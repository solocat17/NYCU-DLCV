import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import zipfile
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import the custom dataset class
from dataloader import ImageClassificationDataset, get_transforms

def get_model(config, model_idx):
    """Create a ResNet101 model with varying configurations for ensemble diversity"""
    
    if config['model'] == 'resnext101_32x8d':
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if config['pretrained'] else None
        model = models.resnext101_32x8d(weights=weights)
    elif config['model'] == 'resnext101_64x4d':
        weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1 if config['pretrained'] else None
        model = models.resnext101_64x4d(weights=weights)
    
    # Get the number of features in the final FC layer
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        fc_attr = 'fc'
    elif hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        fc_attr = 'classifier'
    else:
        # For timm models, the head might have a different name
        if hasattr(model, 'head'):
            in_features = model.head.in_features
            fc_attr = 'head'
        else:
            raise AttributeError("Model structure not recognized")
    
    # Model-specific variations to increase diversity
    if model_idx == 0:
        # Base model with dropout
        setattr(model, fc_attr, nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, config['num_classes'])
        ))
    elif model_idx == 1:
        # Deeper representation with intermediate layer
        setattr(model, fc_attr, nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, config['num_classes'])
        ))
    elif model_idx == 2:
        # Two extra FC layers with batch normalization
        setattr(model, fc_attr, nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, config['num_classes'])
        ))
    elif model_idx == 3:
        # Residual connection in classifier
        inter_features = 1024
        main_path = nn.Sequential(
            nn.Linear(in_features, inter_features),
            nn.ReLU(),
            nn.BatchNorm1d(inter_features),
            nn.Dropout(0.4),
            nn.Linear(inter_features, config['num_classes'])
        )
        
        shortcut = nn.Sequential(
            nn.Linear(in_features, config['num_classes'])
        )
        
        class ResidualFC(nn.Module):
            """Residual fully connected module."""
            def __init__(self, main_path, shortcut):
                super(ResidualFC, self).__init__()
                self.main_path = main_path
                self.shortcut = shortcut
                
            def forward(self, x):
                return self.main_path(x) + self.shortcut(x)
        
        setattr(model, fc_attr, ResidualFC(main_path, shortcut))
    elif model_idx == 4:
        # Multi-head attention approach for features
        class SelfAttentionFC(nn.Module):
            """Self-attention based fully connected module."""
            def __init__(self, in_features, num_classes):
                super(SelfAttentionFC, self).__init__()
                self.query = nn.Linear(in_features, 512)
                self.key = nn.Linear(in_features, 512)
                self.value = nn.Linear(in_features, 512)
                self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
                
            def forward(self, x):
                # Reshape for self-attention if needed
                orig_shape = x.shape
                if len(orig_shape) > 2:
                    x = x.view(orig_shape[0], -1)
                
                # Self-attention mechanism
                q = self.query(x).unsqueeze(1)  # [B, 1, 512]
                k = self.key(x).unsqueeze(1)    # [B, 1, 512]
                v = self.value(x).unsqueeze(1)  # [B, 1, 512]
                
                attn = F.softmax((q @ k.transpose(-2, -1)) / (512 ** 0.5), dim=-1)
                out = (attn @ v).squeeze(1)
                return self.fc(out)
        
        setattr(model, fc_attr, SelfAttentionFC(in_features, config['num_classes']))
    
    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model {model_idx} parameter count: {param_count:,} (limit: 100M)")
    
    if param_count > 100_000_000:
        raise ValueError(f"Model exceeds parameter limit of 100M: {param_count:,}")
    
    return model


def draw_confusion_matrix(all_labels, ensemble_predictions, class_names, save_dir, dataset_type):
    """Draw a confusion matrix for the given true and predicted labels."""


    # For ensemble model
    cm_ensemble = confusion_matrix(all_labels, ensemble_predictions)
    plt.figure(figsize=(55,50))
    sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - Ensemble ({dataset_type} dataset)')
    plt.tight_layout()
    filename = os.path.join(save_dir, f'confusion_matrix_ensemble_{dataset_type}.png')
    plt.savefig(filename)
    plt.close()

def ensemble_predict(models_list, images, device, weights=None, strategy='weighted_avg'):
    """
    Make predictions using an ensemble of models with different aggregation strategies
    
    Args:
        models_list: List of trained models
        images: Batch of images to predict
        device: Device to run inference on
        weights: Optional weights for each model (e.g., validation accuracies)
        strategy: Aggregation strategy ('weighted_avg', 'avg', 'max_conf', 'voting')
        
    Returns:
        Ensemble predictions
    """
    all_outputs = []
    all_probs = []
    
    # Get predictions from each model
    for model in models_list:
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        all_outputs.append(outputs)
        all_probs.append(probs)
    
    # Stack predictions
    stacked_probs = torch.stack(all_probs)
    
    if strategy == 'weighted_avg' and weights is not None:
        # Normalize weights
        weights = torch.tensor(weights).to(device)
        weights = weights / weights.sum()
        
        # Apply weights to each model's predictions
        weighted_probs = stacked_probs * weights.view(-1, 1, 1)
        ensemble_probs = weighted_probs.sum(dim=0)
    
    elif strategy == 'avg':
        # Simple averaging of probabilities
        ensemble_probs = stacked_probs.mean(dim=0)
    
    elif strategy == 'max_conf':
        # Take prediction with highest confidence
        max_conf, _ = torch.max(stacked_probs.max(dim=2)[0], dim=0)
        max_conf_idx = torch.zeros_like(max_conf, dtype=torch.long)
        
        # Find which model had the max confidence for each sample
        for i in range(len(images)):
            max_values = stacked_probs[:, i, :].max(dim=1)[0]
            model_idx = max_values.argmax()
            max_conf_idx[i] = model_idx
            
        # Select probabilities from the most confident model for each sample
        ensemble_probs = torch.zeros_like(all_probs[0])
        for i in range(len(images)):
            ensemble_probs[i] = all_probs[max_conf_idx[i]][i]
    
    elif strategy == 'voting':
        # Hard voting - take the most common class prediction
        preds = []
        for probs in all_probs:
            _, pred = torch.max(probs, dim=1)
            preds.append(pred)
        
        # Stack predictions from all models
        preds = torch.stack(preds)
        
        # Count votes for each class
        ensemble_preds = torch.zeros(images.size(0), dtype=torch.long, device=device)
        
        # For each sample
        for i in range(images.size(0)):
            # Get predictions from all models for this sample
            sample_preds = preds[:, i]
            
            # Count occurrences of each class
            classes, counts = torch.unique(sample_preds, return_counts=True)
            
            # Get class with most votes (or earliest in case of tie)
            max_idx = counts.argmax()
            ensemble_preds[i] = classes[max_idx]
        
        # Convert back to one-hot probabilities
        ensemble_probs = torch.zeros_like(all_probs[0])
        for i in range(images.size(0)):
            ensemble_probs[i, ensemble_preds[i]] = 1.0
    
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")
    
    return ensemble_probs


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate an ensemble of image classifiers')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True, 
                        help='Paths to saved models (specify 5 models)')
    parser.add_argument('--strategy', type=str, default='weighted_avg',
                        choices=['weighted_avg', 'avg', 'max_conf', 'voting'],
                        help='Ensemble prediction strategy')
    args = parser.parse_args()
    
    # Read config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['save_dir'] = os.path.join('saved_models', config['save_dir'])
    
    if len(args.model_paths) != 5:
        raise ValueError("Exactly 5 model paths must be provided for the ensemble")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset and validation dataset
    test_dataset = ImageClassificationDataset(
        root_dir=config['test_dir'],
        transform=get_transforms(is_train=False),
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_dataset = ImageClassificationDataset(
        root_dir=config['val_dir'],
        transform=get_transforms(is_train=False),
        is_test=False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Create and load models
    models_list = []
    val_accs = []
    total_params = 0
    for idx, model_path in enumerate(args.model_paths):
        model = get_model(config, idx)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        params = sum(p.numel() for p in model.parameters())
        total_params += params
        model = model.to(device)
        model.eval()
        models_list.append(model)
        val_acc = checkpoint.get('val_acc', 0)
        val_accs.append(val_acc)
        print(f"Loaded model {idx} from {model_path} - Val acc: {val_acc:.2f}%")
    
    print(f"Total parameters: {total_params} (limit: 100M)")
    print(f"Using ensemble strategy: {args.strategy}")

    # Draw confusion matrix on training and validation dataset with 100 classes
    with torch.no_grad():
        # Training dataset
        all_labels = []
        ensemble_predictions = []
        class_names = [str(i) for i in range(100)]
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            all_labels.extend(labels.numpy())
            ensemble_probs = ensemble_predict(
                models_list, 
                images, 
                device, 
                weights=val_accs if args.strategy == 'weighted_avg' else None,
                strategy=args.strategy
            )
            _, predictions = torch.max(ensemble_probs, dim=1)
            ensemble_predictions.extend(predictions.cpu().numpy())
        draw_confusion_matrix(all_labels, ensemble_predictions, class_names, config['save_dir'], 'val')

    # Ensemble prediction for test dataset
    all_probs = []
    image_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            
            # Get ensemble predictions
            ensemble_probs = ensemble_predict(
                models_list, 
                images, 
                device, 
                weights=val_accs if args.strategy == 'weighted_avg' else None,
                strategy=args.strategy
            )
            
            all_probs.append(ensemble_probs.cpu())
            
            # Save image paths
            if isinstance(paths[0], torch.Tensor):
                paths = [p.item() if isinstance(p, torch.Tensor) else p for p in paths]
            image_paths.extend(paths)
    
    # Concatenate all batch probabilities
    all_probs = torch.cat(all_probs, dim=0)
    
    # Get predicted classes from ensemble
    _, predictions = torch.max(all_probs, dim=1)
    predictions = predictions.numpy()
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'image_path': image_paths,
        'pred_label': predictions
    })
    
    # Extract filename from path
    submission['image_name'] = submission['image_path'].apply(lambda x: os.path.basename(x))
    submission['image_name'] = submission['image_name'].str.replace('.jpg', '')
    
    # Save submission
    submission_path = os.path.join(config['save_dir'], 'prediction.csv')
    submission[['image_name', 'pred_label']].to_csv(submission_path, index=False)
    
    # Zip the submission file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with zipfile.ZipFile(f'{config["save_dir"]}/submission_{timestamp}_{args.strategy}.zip', 'w') as z:
        z.write(submission_path, 'prediction.csv')
    
    print(f"Saved submission to {submission_path}")
    print(f"Zipped submission to {config['save_dir']}/submission_{timestamp}_{args.strategy}.zip")
    
if __name__ == '__main__':
    main()
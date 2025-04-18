import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# Import custom dataset and transforms
from cutmixdataloader import get_data_loaders

def parse_arguments():
    """Parse command-line arguments and configuration file."""
    parser = argparse.ArgumentParser(description='Train an ensemble of SE-ResNeXt image classifiers')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--models_to_train', type=str, required=False, 
                        help='Models to train (comma-separated indices, e.g., 0,1,2)')
    args = parser.parse_args()
    
    # Read config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Parse models_to_train argument
    config['models_to_train'] = [int(i) for i in args.models_to_train.split(',')] if args.models_to_train else [0, 1, 2, 3, 4]
    
    # Ensure save directory is properly set
    config['save_dir'] = os.path.join('saved_models', config['save_dir'])

    return config


def get_model(config, model_idx):
    """
    Create a SE / SK -ResNeXt model with varying configurations for ensemble diversity.
    
    Args:
        config (dict): Configuration parameters.
        model_idx (int): Index of the model in the ensemble (determines configuration).
        
    Returns:
        torch.nn.Module: The configured model.
    """
    # Create SE-ResNeXt with pre-trained weights
    if config['model'] == 'seresnext':
        model = timm.create_model('seresnext26d_32x4d', pretrained=config['pretrained'])
    elif config['model'] == 'skresnext':
        model = timm.create_model('skresnext50_32x4d', pretrained=config['pretrained'])
    else:
        raise ValueError(f"Unknown model type: {config['model']}")
    
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


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (cuda/cpu)
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for data in progress_bar:
        # Check if we're using CutMix
        if len(data) == 4:
            # CutMix data
            images, targets_a, targets_b, lam = data
            # show target_a, target_b, lam
            # print(targets_a, targets_b, lam)
            images = images.to(device)
            # targets_a, targets_b are None, so we need to convert them to tensor
            targets_a = torch.tensor(targets_a).to(device)
            targets_b = torch.tensor(targets_b).to(device)
            # targets_a, targets_b = targets_a.to(device), targets_b.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # CutMix loss
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
            # For accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += targets_a.size(0)
            # For mixed samples, count a prediction as correct if it matches the dominant label
            # (determined by lambda)
            correct_a = (predicted == targets_a).float()
            correct_b = (predicted == targets_b).float()
            correct += (lam * correct_a + (1 - lam) * correct_b).sum().item()
        else:
            # Regular data
            images, targets = data
            images, targets = images.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # For accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / total, 
            'acc': 100 * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    Validate the model performance.
    
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on (cuda/cpu)
        
    Returns:
        tuple: (val_loss, val_accuracy)
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def train_one_model(config, model_idx):
    """
    Train a single model with the given configuration and model index.
    
    Args:
        config (dict): Configuration parameters.
        model_idx (int): Index of the model in the ensemble.
    """

    # Set device
    device = torch.device(f"cuda:{model_idx % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    print(f"Training model {model_idx} on device: {device}")
    
    # Create model save directory if it doesn't exist
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load datasets
    train_loader, val_loader, _ = get_data_loaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_cutmix=config.get('use_cutmix', True)
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = get_model(config, model_idx)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Update learning rate if scheduler is enabled
        if config.get('use_scheduler', False):
            scheduler.step(val_loss)

        model_name = f"{config['model']}_{model_idx}"
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(config['save_dir'], f'best_model_{model_name}.pth'))
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % config.get('checkpoint_interval', 5) == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(config['save_dir'], f'checkpoint_model_{model_name}_epoch_{epoch+1}.pth'))
    
    # Calculate training time
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Model {model_idx} Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'Model {model_idx} Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], f'training_curves_model_{model_name}.png'))
    plt.close()

    # save loss and accuracy to file
    with open(os.path.join(config['save_dir'], f'training_curves_model_{model_name}.txt'), 'w') as f:
        f.write(f"Train Loss: {train_losses}\n")
        f.write(f"Val Loss: {val_losses}\n")
        f.write(f"Train Acc: {train_accs}\n")
        f.write(f"Val Acc: {val_accs}\n")


def main():
    # Parse arguments
    config = parse_arguments()
    
    # Set up distributed training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for distributed training")
        
        # Set multiprocessing start method (important for CUDA)
        mp.set_start_method('spawn', force=True)

        # Parallelize training of ensemble models
        processes = []
        for i in range(config['ensemble_models']):
            if i not in config['models_to_train']:
                print(f"Skipping model {i} as it is not in models_to_train list")
                continue
                
            # Each process uses a specific GPU
            p = mp.Process(target=train_one_model, args=(config, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # Sequential training if only one GPU is available
        print("Training models sequentially (only one GPU found)")
        for i in range(config['ensemble_models']):
            if i not in config['models_to_train']:
                print(f"Skipping model {i} as it is not in models_to_train list")
                continue
            train_one_model(config, i)

    print("Training completed")

if __name__ == '__main__':
    main()
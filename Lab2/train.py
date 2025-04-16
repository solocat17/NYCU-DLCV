import argparse
import os
import torch
import yaml
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import wandb  # Add wandb import
from src.dataloader import get_dataloaders
from src.model import DigitDetector
from src.trainer import Trainer
from src.utils import save_checkpoint, load_checkpoint, calculate_mAP
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train Digit Recognition Model")
    parser.add_argument('--config', type=str, default='configs/default.yml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='dlcv_lab2', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')
    return parser.parse_args()

def copy_source_code(src_dir, dst_dir):
    """
    Copy source code files from src_dir to dst_dir.
    """
    dst_dir = os.path.join(dst_dir, 'source')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file in ['train.py', 'test.py']:
        src_file = os.path.join('.', file)
        dst_file = os.path.join(dst_dir, file)
        with open(src_file, 'r') as fsrc:
            with open(dst_file, 'w') as fdst:
                fdst.write(fsrc.read())
    # Copy the entire src directory
    src_dir = os.path.join('.', 'src')
    dst_dir = os.path.join(dst_dir, 'src')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for file in os.listdir(src_dir):
        if file.endswith('.py'):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            with open(src_file, 'r') as fsrc:
                with open(dst_file, 'w') as fdst:
                    fdst.write(fsrc.read())


def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    exp_dir = os.path.join('results', args.exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    vis_dir = os.path.join(exp_dir, 'visualizations')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(exp_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    
    # Copy source code
    copy_source_code('.', exp_dir)

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            config={
                **config,
                "data_dir": args.data_dir,
            }
        )
        # Log the code files to wandb
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    model = DigitDetector(num_classes=11)  # 10 digits (0-9) + background
    
    # Create optimizer
    if config['optimizer']['name'] == 'sgd':
        optimizer = SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay']
        )
    elif config['optimizer']['name'] == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']['name']}")
    
    # Create learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config['scheduler']['step_size'],
        gamma=config['scheduler']['gamma']
    )
    
    # Watch model with wandb if enabled
    if args.wandb:
        wandb.watch(model, log="all", log_freq=10)
    
    # Create trainer
    trainer = Trainer(model, optimizer, scheduler, device, use_wandb=args.wandb)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_map = 0.0
    train_losses = []
    val_maps = []

    for epoch in range(num_epochs):
        # Train for one epoch
        train_metrics = trainer.train_one_epoch(train_loader, epoch)
        train_loss = train_metrics['loss']
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_loss, None, checkpoint_dir
        )

        # Evaluate on validation set
        val_predictions = trainer.evaluate(valid_loader)

        # Calculate mAP
        ground_truth = [
            {'boxes': batch['boxes'], 'categories': batch['categories']} 
            for batch in valid_loader.dataset
        ]
        iou_thresholds = np.arange(0.3, 1.0, 0.05)
        val_map = calculate_mAP(val_predictions, ground_truth, iou_thresholds)
        val_maps.append(val_map)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Validation mAP: {val_map:.4f}")
        
        # Log metrics to wandb
        if args.wandb:
            wandb_metrics = {
                'train/loss': train_loss,
                'train/loss_classifier': train_metrics['loss_classifier'],
                'train/loss_box_reg': train_metrics['loss_box_reg'],
                'train/loss_objectness': train_metrics['loss_objectness'],
                'train/loss_rpn_box_reg': train_metrics['loss_rpn_box_reg'],
                'val/mAP': val_map,
                'lr': optimizer.param_groups[0]['lr']
            }
            wandb.log(wandb_metrics, step=epoch)
        
        # Save best model based on mAP
        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation mAP: {val_map:.4f}")
            
            # Log best model as artifact
            if args.wandb:
                model_artifact = wandb.Artifact(f"{args.exp_name}_best_model", type="model")
                model_artifact.add_file(best_model_path)
                wandb.log_artifact(model_artifact)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_maps, label='Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Training and Validation mAP')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(vis_dir, 'loss_mAP_curve.png')
    plt.savefig(loss_curve_path)
    plt.close()
    
    # Log the figure to wandb
    if args.wandb:
        wandb.log({"train_val_curve": wandb.Image(loss_curve_path)})
        
        # Finish the wandb run
        wandb.finish()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
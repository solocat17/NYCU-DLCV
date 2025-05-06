import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
import argparse
from pathlib import Path
import json

from src.dataloader import get_dataloader
from src.model import get_maskrcnn_model, get_maskrcnn_model_fpnv2, get_maskrcnn_resnext101_fpn, get_maskrcnn_resnext101_unet
from src.utils import format_predictions, save_predictions, copy_source_code
from src.inference import inference
from src.trainer import train_one_epoch, evaluate, save_checkpoint, load_checkpoint, log_metrics


def visualize_loss(train_loss, val_loss, val_metrics, output_dir):
    """
    Visualize training and validation loss over epochs.
    
    Args:
        train_loss (list): List of training loss values.
        val_loss (list): List of validation loss values.
        val_metrics (list): List of validation metrics.
        (metrics = {
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_mask': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        })
        output_dir (str): Directory to save the plots.
    """
    metrics = {
        'loss_classifier': [],
        'loss_box_reg': [],
        'loss_mask': [],
        'loss_objectness': [],
        'loss_rpn_box_reg': [],
        'val_time': [],
        'total_loss': []
    }
    for m in val_metrics:
        metrics['loss_classifier'].append(m['loss_classifier'])
        metrics['loss_box_reg'].append(m['loss_box_reg'])
        metrics['loss_mask'].append(m['loss_mask'])
        metrics['loss_objectness'].append(m['loss_objectness'])
        metrics['loss_rpn_box_reg'].append(m['loss_rpn_box_reg'])
        metrics['val_time'].append(m['val_time'])
        metrics['total_loss'].append(
            m['loss_classifier'] + m['loss_box_reg'] + m['loss_mask'] +
            m['loss_objectness'] + m['loss_rpn_box_reg']
        )

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation metrics
    plt.subplot(1, 2, 2)
    for metric_name, metric_values in metrics.items():
        if metric_name != 'val_time':
            plt.plot(metric_values, label=metric_name)
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

def train_model(data_root, output_dir, num_epochs=30, batch_size=2, lr=0.001, 
                weight_decay=1e-4, model_type='fpn', num_classes=5, 
                num_workers=4, device=None, val_interval=1,
                checkpoint_interval=5, resume_from=None, height=1024, width=1024,
                validation_split=0.2, seed=42, disable_spatial_transforms=False):
    """
    Train a Mask R-CNN model for instance segmentation.
    
    Args:
        data_root (str): Root directory for the dataset.
        output_dir (str): Directory to save checkpoints and logs.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Initial learning rate.
        weight_decay (float): Weight decay for optimizer.
        model_type (str): Type of model to use ('fpn' or 'fpnv2').
        num_classes (int): Number of classes (including background).
        num_workers (int): Number of workers for data loading.
        device (torch.device): Device to train on.
        val_interval (int): Validation frequency (in epochs).
        checkpoint_interval (int): Checkpoint saving frequency (in epochs).
        resume_from (str): Path to checkpoint to resume training from.
        height (int): Image height for training.
        width (int): Image width for training.
        validation_split (float): Fraction of the dataset to use for validation.
        seed (int): Random seed for reproducibility.
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_source_code('.', output_dir)
    
    # Create log file
    log_file = output_dir / 'training_log.txt'
    
    # Create data loaders with validation split
    train_loader, val_loader = get_dataloader(
        root_dir=data_root,
        batch_size=batch_size,
        split='train',
        num_workers=num_workers,
        height=height,
        width=width,
        disable_spatial_transforms=disable_spatial_transforms,
        validation_split=validation_split,
        seed=seed
    )
    
    # Create model
    print(f"Creating model of type {model_type} with {num_classes} classes")
    if model_type == 'fpn':
        model = get_maskrcnn_model(num_classes=num_classes)
    elif model_type == 'fpnv2':
        model = get_maskrcnn_model_fpnv2(num_classes=num_classes)
    elif model_type == 'resnext_fpn':
        model = get_maskrcnn_resnext101_fpn(num_classes=num_classes)
    elif model_type == 'resnext_unet':
        model = get_maskrcnn_resnext101_unet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params / 1e6:.2f} M parameters")
    if (num_params > 200e6):
        print("Model size exceeds 200M parameters, consider using a smaller model.")
        return

    # Move model to device
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, 
                                  verbose=True, min_lr=1e-6)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
    
    # Create dict to store training stats
    stats = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'learning_rates': []
    }
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(start_epoch, num_epochs):
        # TRAINING PHASE
        train_loss, epoch_time = train_one_epoch(model, train_loader, optimizer, device)
        
        # Log training stats
        current_lr = optimizer.param_groups[0]['lr']
        log_metrics(
            epoch=epoch+1,
            train_loss=train_loss,
            lr=current_lr,
            epoch_time=epoch_time,
            log_file=log_file
        )
        
        # Update stats
        stats['train_loss'].append(train_loss)
        stats['learning_rates'].append(current_lr)
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_loss,
                output_dir=output_dir,
                filename="best_model.pth"
            )
            print(f"Saved new best model with training loss: {train_loss:.4f}")

        # VALIDATION PHASE
        if (epoch + 1) % val_interval == 0 and val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, device)
            
            # Log validation results
            log_metrics(
                epoch=epoch+1,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
                lr=current_lr,
                log_file=log_file
            )
            
            # Update stats
            stats['val_loss'].append(val_loss)
            stats['val_metrics'].append(val_metrics)
            
            # Update learning rate scheduler based on validation loss
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_loss=best_loss,
                    output_dir=output_dir,
                    filename="best_model.pth"
                )
                print(f"Saved new best model with validation loss: {val_loss:.4f}")
        elif isinstance(scheduler, ReduceLROnPlateau):
            # Update learning rate scheduler based on training loss if no validation
            scheduler.step(train_loss)
        
        # Save regular checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_loss,
                output_dir=output_dir,
                filename=f"checkpoint_epoch_{epoch+1}.pth"
            )
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs-1,
        best_loss=best_loss,
        output_dir=output_dir,
        filename="final_model.pth"
    )
    
    # Save training stats
    stats_file = output_dir / 'training_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)

    # Visualize loss
    visualize_loss(stats['train_loss'], stats['val_loss'], stats['val_metrics'], output_dir)
    
    print(f"Training completed. Model saved to {output_dir}")
    return model, stats


def main():
    begin_time = time.time()
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for cell instance segmentation")
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='.', help='Root directory for the dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--height', type=int, default=1024, help='Image height for training')
    parser.add_argument('--width', type=int, default=1024, help='Image width for training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation frequency (epochs)')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Checkpoint saving frequency (epochs)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='fpn', choices=['fpn', 'fpnv2', 'resnext_fpn', 'resnext_unet'],
                        help='Type of Mask R-CNN model')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes (including background)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    
    # Disable spatial transforms
    parser.add_argument('--disable_spatial_transforms', action='store_true', 
                        help='Disable spatial transformations in data augmentation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure logging
    logging_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            }
        }
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command line arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    try:
        # Train model
        model, stats = train_model(
            data_root=args.data_root,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_type=args.model_type,
            num_classes=args.num_classes,
            num_workers=args.num_workers,
            device=device,
            val_interval=args.val_interval,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume,
            height=args.height,
            width=args.width,
            validation_split=args.validation_split,
            seed=args.seed,
            disable_spatial_transforms=args.disable_spatial_transforms
        )
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        # Save error information
        with open(os.path.join(args.output_dir, 'error_log.txt'), 'w') as f:
            f.write(traceback.format_exc())

    end_time = time.time()
    elapsed_time = end_time - begin_time
    elapsed_hours, elapsed_minutes, elapsed_seconds = (
        elapsed_time // 3600,
        (elapsed_time % 3600) // 60,
        elapsed_time % 60
    )
    print(f"Total training time: {int(elapsed_hours)}h {int(elapsed_minutes)}m {int(elapsed_seconds)}s")


if __name__ == "__main__":
    main()
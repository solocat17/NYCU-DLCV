import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer for model parameters.
        device (torch.device): Device to train on.
        
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    # count_type_of_label = {}


    for images, targets in progress_bar:
        # Move images and targets to device
        # img is a np.ndarray, convert to torch.Tensor
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in t.items()} for t in targets]

        # label = targets[0]['labels']
        # for l in label:
        #     count_type_of_label[l.item()] = count_type_of_label.get(l.item(), 0) + 1

        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        result = model(images, targets)
        
        # Calculate total loss based on the output type
        if isinstance(result, dict):
            # Dictionary of losses
            loss_dict = result
            losses = sum(loss for loss in loss_dict.values())
        elif isinstance(result, list):
            # List of losses
            losses = sum(result)
        else:
            # Single loss value
            losses = result
        
        # Backward pass and optimize
        losses.backward()
        optimizer.step()
        
        # Update progress bar
        epoch_loss += losses.item()
        progress_bar.set_postfix({"Loss": losses.item()})
    
    # # Print label counts
    # print("Label counts in this batch:")
    # for label, count in count_type_of_label.items():
    #     print(f"Label {label}: {count} instances")

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    epoch_time = time.time() - start_time
    
    return avg_epoch_loss, epoch_time


def evaluate(model, dataloader, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Device to evaluate on.
        
    Returns:
        float: Average validation loss.
        dict: Dictionary containing additional evaluation metrics.
    """
    # Mask R-CNN needs to be in training mode to compute losses, even during validation
    model.train()
    val_loss = 0.0
    val_start_time = time.time()
    
    # Additional metrics to track - these might not all be used depending on model output
    metrics = {
        'loss_classifier': 0.0,
        'loss_box_reg': 0.0,
        'loss_mask': 0.0,
        'loss_objectness': 0.0,
        'loss_rpn_box_reg': 0.0
    }
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for images, targets in progress_bar:
            # Move images and targets to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in t.items()} for t in targets]
            
            # Forward pass
            result = model(images, targets)
            
            # Handle different return types in evaluation mode
            if isinstance(result, dict):
                # It's a dictionary of losses
                loss_dict = result
                losses = sum(loss for loss in loss_dict.values())
                
                # Update additional metrics
                for k, v in loss_dict.items():
                    if k in metrics:
                        metrics[k] += v.item()
            elif isinstance(result, list):
                # It's a list of losses
                losses = sum(result)
                
                # If we have a list, we can't update individual metrics
                # because we don't know which loss corresponds to which metric
            else:
                # It's a single loss value
                losses = result
            
            # Update progress bar
            val_loss += losses.item()
            progress_bar.set_postfix({"Val Loss": losses.item()})
    
    # Calculate average loss and metrics
    avg_val_loss = val_loss / len(dataloader)
    val_time = time.time() - val_start_time
    
    # Normalize metrics
    for k in metrics:
        metrics[k] /= len(dataloader)
    
    # Add validation time to metrics
    metrics['val_time'] = val_time
    
    return avg_val_loss, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, output_dir, filename="checkpoint.pth"):
    """
    Save a model checkpoint.
    
    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer state to save.
        scheduler: The scheduler state to save.
        epoch (int): Current epoch number.
        best_loss (float): Best validation loss so far.
        output_dir (str or Path): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, output_dir / filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device=None):
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load weights into.
        optimizer (Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (torch.device, optional): Device to load the checkpoint onto.
        
    Returns:
        dict: Checkpoint data with keys 'epoch', 'best_loss', etc.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def log_metrics(epoch, train_loss, val_loss=None, metrics=None, lr=None, epoch_time=None, log_file=None):
    """
    Log training and validation metrics.
    
    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        val_loss (float, optional): Validation loss.
        metrics (dict, optional): Additional metrics to log.
        lr (float, optional): Current learning rate.
        epoch_time (float, optional): Time taken for the epoch.
        log_file (str or Path, optional): Path to log file. If None, logs to console only.
    """
    log_str = f"Epoch {epoch} completed. Train Loss: {train_loss:.4f}"
    
    if val_loss is not None:
        log_str += f", Val Loss: {val_loss:.4f}"
    
    if lr is not None:
        log_str += f", LR: {lr:.6f}"
    
    if epoch_time is not None:
        log_str += f", Time: {epoch_time:.2f}s"
    
    # Add additional metrics if provided
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                log_str += f", {key}: {value:.4f}"
            else:
                log_str += f", {key}: {value}"
    
    # Print to console
    print(log_str)
    
    # Write to log file if provided
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')
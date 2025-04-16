import torch
from tqdm import tqdm
import os
import time
import datetime
import numpy as np
from collections import defaultdict
import wandb

class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device='cuda', use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb
        self.model.to(self.device)
        
    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        
        epoch_loss = 0.0
        epoch_loss_classifier = 0.0
        epoch_loss_box_reg = 0.0
        epoch_loss_objectness = 0.0
        epoch_loss_rpn_box_reg = 0.0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = list(image.to(self.device) for image in batch['images'])
            
            targets = self.model.convert_targets(
                batch['boxes'], 
                batch['categories']
            )
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Reduce losses over all GPUs for logging purposes
            loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            
            epoch_loss += losses_reduced
            epoch_loss_classifier += loss_dict_reduced.get('loss_classifier', 0)
            epoch_loss_box_reg += loss_dict_reduced.get('loss_box_reg', 0)
            epoch_loss_objectness += loss_dict_reduced.get('loss_objectness', 0)
            epoch_loss_rpn_box_reg += loss_dict_reduced.get('loss_rpn_box_reg', 0)
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses_reduced:.4f}"
            })
            
            # Log batch-level metrics to wandb
            if self.use_wandb and batch_idx % 20 == 0:  # Log every 20 batches
                step = epoch * len(data_loader) + batch_idx
                wandb.log({
                    'batch/loss': losses_reduced,
                    'batch/loss_classifier': loss_dict_reduced.get('loss_classifier', 0),
                    'batch/loss_box_reg': loss_dict_reduced.get('loss_box_reg', 0),
                    'batch/loss_objectness': loss_dict_reduced.get('loss_objectness', 0),
                    'batch/loss_rpn_box_reg': loss_dict_reduced.get('loss_rpn_box_reg', 0),
                }, step=step)
            
        if self.scheduler:
            self.scheduler.step()
            
        num_batches = len(data_loader)
        epoch_loss /= num_batches
        epoch_loss_classifier /= num_batches
        epoch_loss_box_reg /= num_batches
        epoch_loss_objectness /= num_batches
        epoch_loss_rpn_box_reg /= num_batches
        
        # At the end of train_one_epoch
        print(f"Epoch {epoch+1} loss breakdown:")
        print(f"  Classifier loss: {epoch_loss_classifier:.4f}")
        print(f"  Box reg loss: {epoch_loss_box_reg:.4f}")
        print(f"  Objectness loss: {epoch_loss_objectness:.4f}")
        print(f"  RPN box reg loss: {epoch_loss_rpn_box_reg:.4f}")

        return {
            'loss': epoch_loss,
            'loss_classifier': epoch_loss_classifier,
            'loss_box_reg': epoch_loss_box_reg,
            'loss_objectness': epoch_loss_objectness,
            'loss_rpn_box_reg': epoch_loss_rpn_box_reg
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        
        all_predictions = []
        
        progress_bar = tqdm(data_loader, desc="Validation")
        
        for batch in progress_bar:
            images = list(image.to(self.device) for image in batch['images'])
            image_ids = batch['image_ids']
            
            # Get predictions (model in eval mode without targets returns predictions)
            predictions = self.model(images)
            
            # Store predictions for mAP calculation
            for i, pred in enumerate(predictions):
                pred_dict = {
                    'image_id': image_ids[i].item(),
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                }
                all_predictions.append(pred_dict)
                
                # Log sample predictions to wandb (only a few for visualization)
                if self.use_wandb and i % 50 == 0 and len(pred['boxes']) > 0:
                    try:
                        # Get the original image
                        image = images[i].cpu().permute(1, 2, 0).numpy()
                        
                        # Create a wandb bounding box visualization
                        boxes_data = []
                        for box_idx in range(min(5, len(pred['boxes']))):  # Log up to 5 boxes per image
                            box = pred['boxes'][box_idx].tolist()
                            label = pred['labels'][box_idx].item()
                            score = pred['scores'][box_idx].item()
                            
                            # Create wandb box data
                            box_data = {
                                "position": {
                                    "minX": box[0],
                                    "minY": box[1],
                                    "maxX": box[2],
                                    "maxY": box[3]
                                },
                                "class_id": int(label),
                                "box_caption": f"Digit: {label-1 if label > 0 else 'background'}, {score:.2f}",
                                "scores": {"confidence": score}
                            }
                            boxes_data.append(box_data)
                        
                        if boxes_data:
                            wandb.log({
                                f"val_predictions/image_{image_ids[i].item()}": wandb.Image(
                                    image,
                                    boxes={
                                        "predictions": {
                                            "box_data": boxes_data,
                                            "class_labels": {i: str(i-1) if i > 0 else "background" for i in range(11)}
                                        }
                                    }
                                )
                            })
                    except Exception as e:
                        print(f"Error logging prediction visualization to wandb: {e}")
        
            targets = self.model.convert_targets(
                batch['boxes'], 
                batch['categories']
            )
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        return all_predictions
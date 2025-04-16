import os
import torch
import numpy as np
from collections import defaultdict
from torchvision.ops import box_iou

def save_checkpoint(model, optimizer, scheduler, epoch, loss, metrics, checkpoint_dir):
    """
    Save a checkpoint of the model state
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The learning rate scheduler
        epoch: Current epoch number
        loss: Current loss value
        metrics: Dictionary of metrics
        checkpoint_dir: Directory to save checkpoints
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add metrics if available
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load a checkpoint and restore model state
    
    Args:
        model: The model to restore
        optimizer: The optimizer to restore
        scheduler: The learning rate scheduler to restore
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        model: Restored model
        optimizer: Restored optimizer
        scheduler: Restored scheduler
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
        metrics: Metrics from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    # Restore scheduler if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get metrics if available
    metrics = checkpoint.get('metrics', None)
    
    return model, optimizer, scheduler, epoch, loss, metrics

def calculate_ap(class_preds, class_targets, iou_threshold):
    """
    Calculate Average Precision at a specific IoU threshold for a class
    
    Args:
        class_preds: List of prediction dictionaries for this class
        class_targets: List of target dictionaries for this class
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        Average Precision at the specified IoU threshold
    """
    # Handle edge cases
    if not class_preds or not class_targets:
        return 0.0
    
    # Count total number of ground truth boxes
    num_gt_boxes = sum(len(target['boxes']) for target in class_targets)
    
    if num_gt_boxes == 0:
        return 0.0  # No ground truth boxes means AP is 0
    
    # Prepare ground truth data structure
    gt_boxes_by_image = _prepare_ground_truth_data(class_targets)
    
    # Collect and sort all predictions
    all_predictions = _collect_predictions(class_preds)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Evaluate each prediction
    all_scores, all_true_positives = _evaluate_predictions(all_predictions, gt_boxes_by_image, iou_threshold)
    
    # If no predictions, return 0
    if not all_scores:
        return 0.0
    
    # Convert to numpy arrays for easier calculation
    scores = np.array(all_scores)
    true_positives = np.array(all_true_positives)
    
    # Sort by score (descending)
    sorted_indices = np.argsort(-scores)
    true_positives = true_positives[sorted_indices]
    
    # Compute precision-recall curve
    precision, recall = _compute_precision_recall(true_positives, num_gt_boxes)
    
    # Compute AP using the rectangle rule
    ap = _calculate_ap_from_precision_recall(precision, recall)
    
    return ap


def _prepare_ground_truth_data(class_targets):
    """Prepare ground truth data indexed by image index"""
    gt_boxes_by_image = {}
    for target in class_targets:
        img_idx = target['image_idx']
        gt_boxes_by_image[img_idx] = {
            'boxes': target['boxes'],
            'detected': torch.zeros(len(target['boxes']), dtype=torch.bool)
        }
    return gt_boxes_by_image


def _collect_predictions(class_preds):
    """Collect all predictions across all images"""
    all_predictions = []
    for pred in class_preds:
        img_idx = pred['image_idx']
        boxes = pred['boxes']
        scores = pred['scores']
        
        for box_idx in range(len(boxes)):
            all_predictions.append({
                'box': boxes[box_idx],
                'score': scores[box_idx],
                'img_idx': img_idx
            })
    return all_predictions


def _evaluate_predictions(all_predictions, gt_boxes_by_image, iou_threshold):
    """Evaluate each prediction and determine if it's a true positive"""
    all_scores = []
    all_true_positives = []
    
    for pred in all_predictions:
        img_idx = pred['img_idx']
        box = pred['box']
        score = pred['score']
        
        all_scores.append(
            score.item() if isinstance(score, torch.Tensor) else score
        )
        
        # Check if this image has any ground truth
        if (img_idx not in gt_boxes_by_image or
                len(gt_boxes_by_image[img_idx]['boxes']) == 0):
            all_true_positives.append(0)  # False positive
            continue
        
        # Calculate IoU with all ground truth boxes in this image
        gt_boxes = gt_boxes_by_image[img_idx]['boxes']
        if len(gt_boxes) == 0:
            all_true_positives.append(0)  # False positive
            continue
        
        # If we only have one box, reshape to ensure proper dimensions
        if len(box.shape) == 1:
            box = box.unsqueeze(0)
        
        ious = box_iou(box, gt_boxes)[0]  # [0] to get the first (only) row
        
        # Find best matching ground truth box
        max_iou, max_idx = torch.max(ious, dim=0)
        
        # Check if the IoU is above threshold and the GT box hasn't been detected yet
        if (max_iou >= iou_threshold and
                not gt_boxes_by_image[img_idx]['detected'][max_idx]):
            all_true_positives.append(1)  # True positive
            gt_boxes_by_image[img_idx]['detected'][max_idx] = True
        else:
            all_true_positives.append(0)  # False positive
    
    return all_scores, all_true_positives


def _compute_precision_recall(true_positives, num_gt_boxes):
    """Compute precision and recall values"""
    # Compute cumulative true positives
    cum_true_positives = np.cumsum(true_positives)
    
    # Compute precision and recall
    precision = cum_true_positives / np.arange(1, len(cum_true_positives) + 1)
    recall = cum_true_positives / num_gt_boxes
    
    # Add sentinel values for calculation
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    return precision, recall


def _calculate_ap_from_precision_recall(precision, recall):
    """Calculate AP from precision-recall curve using the rectangle rule"""
    # Find points where recall changes
    recall_changes = np.where(recall[1:] != recall[:-1])[0] + 1
    
    # Calculate AP using the rectangle rule
    ap = np.sum(
        (recall[recall_changes] - recall[recall_changes - 1]) * precision[recall_changes]
    )
    
    return ap


def calculate_mAP(all_predictions, all_targets, iou_thresholds):
    """
    Calculate mean Average Precision (mAP) for object detection
    
    Args:
        all_predictions: List of prediction dictionaries (each containing 'boxes', 'scores', 'labels')
        all_targets: List of target dictionaries (each containing 'boxes', 'labels')
        iou_thresholds: List of IoU thresholds to evaluate
    
    Returns:
        Mean Average Precision (mAP) across all classes and IoU thresholds
    """
    # Check if predictions or targets are empty
    if not all_predictions or not all_targets:
        return 0.0
    
    # Group predictions and targets by class
    class_predictions, class_targets = _group_by_class(all_predictions, all_targets)
    
    # Calculate AP for each class and IoU threshold
    average_precisions = _calculate_class_aps(class_predictions, class_targets, iou_thresholds)
    
    # Calculate mAP (mean of APs across all classes)
    mAP = np.mean(average_precisions) if average_precisions else 0.0
    
    return mAP


def _group_by_class(all_predictions, all_targets):
    """Group predictions and targets by class"""
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    for i, (preds, targets) in enumerate(zip(all_predictions, all_targets)):
        pred_boxes = preds['boxes']
        pred_scores = preds['scores']
        pred_labels = preds['labels']
        
        gt_boxes = targets['boxes']
        gt_labels = targets['categories']
        
        # Skip if no predictions or targets
        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            continue
        
        # Get unique labels from both predictions and targets
        unique_labels = torch.unique(
            torch.cat([pred_labels, gt_labels])
            if len(pred_labels) > 0 and len(gt_labels) > 0
            else (pred_labels if len(pred_labels) > 0 else gt_labels)
        )
        
        # Process each class
        for label in unique_labels:
            label_item = label.item()
            
            # Skip background class if present
            if label_item == 0:
                continue
            
            # Add predictions for this class
            _add_class_predictions(class_predictions, label_item, i, pred_boxes, pred_scores, pred_labels)
            
            # Add targets for this class
            _add_class_targets(class_targets, label_item, i, gt_boxes, gt_labels)
    
    return class_predictions, class_targets


def _add_class_predictions(class_predictions, label_item, image_idx, pred_boxes, pred_scores, pred_labels):
    """Add predictions for a specific class to the class_predictions dictionary"""
    class_mask = pred_labels == label_item
    class_predictions[label_item].append({
        'boxes': (
            pred_boxes[class_mask]
            if len(pred_boxes) > 0 else torch.empty((0, 4))
        ),
        'scores': (
            pred_scores[class_mask]
            if len(pred_scores) > 0 else torch.empty(0)
        ),
        'image_idx': image_idx
    })


def _add_class_targets(class_targets, label_item, image_idx, gt_boxes, gt_labels):
    """Add targets for a specific class to the class_targets dictionary"""
    target_mask = gt_labels == label_item
    class_targets[label_item].append({
        'boxes': (
            gt_boxes[target_mask]
            if len(gt_boxes) > 0 else torch.empty((0, 4))
        ),
        'image_idx': image_idx
    })


def _calculate_class_aps(class_predictions, class_targets, iou_thresholds):
    """Calculate AP for each class across all IoU thresholds"""
    average_precisions = []
    
    # Iterate over classes
    for class_id in class_predictions.keys():
        # Calculate AP at different IoU thresholds
        aps_for_class = []
        
        for iou_threshold in iou_thresholds:
            ap = calculate_ap(
                class_predictions[class_id],
                class_targets[class_id],
                iou_threshold
            )
            aps_for_class.append(ap)
        
        # Average AP across IoU thresholds (AP@[.3:.95])
        class_ap = np.mean(aps_for_class) if aps_for_class else 0.0
        average_precisions.append(class_ap)

        print(f"Class {class_id}: AP@[.3:.95] = {class_ap:.4f}")

    print(f"Mean Average Precision (mAP) = {np.mean(average_precisions):.4f}")
    
    return average_precisions
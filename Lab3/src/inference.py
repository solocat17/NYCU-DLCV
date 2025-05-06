import torch
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import cv2

from src.dataloader import get_dataloader
from src.model import get_maskrcnn_model, get_maskrcnn_model_fpnv2, get_maskrcnn_resnext101_fpn, get_maskrcnn_resnext101_unet
from src.model import MaskRCNNWrapper, MaskRCNNResNeXt101Wrapper
from src.utils import format_predictions, save_predictions


def resize_predictions_to_original(predictions, model_input_height, model_input_width):
    """
    Resize predictions from model input size to original image size.
    
    Args:
        predictions (list): List of prediction dictionaries
        model_input_height (int): Height of images input to the model
        model_input_width (int): Width of images input to the model
        
    Returns:
        list: Resized predictions
    """
    resized_predictions = []
    
    for pred in predictions:
        orig_height = pred.get('orig_height')
        orig_width = pred.get('orig_width')
        
        # Skip if original dimensions are not available
        if orig_height is None or orig_width is None:
            resized_predictions.append(pred)
            continue
        
        # Calculate scale factors
        scale_y = orig_height / model_input_height
        scale_x = orig_width / model_input_width
        
        # Scale bounding boxes
        if 'boxes' in pred and len(pred['boxes']) > 0:
            # Scale boxes [x1, y1, x2, y2]
            scaled_boxes = pred['boxes'].clone()
            scaled_boxes[:, 0] *= scale_x  # x1
            scaled_boxes[:, 1] *= scale_y  # y1
            scaled_boxes[:, 2] *= scale_x  # x2
            scaled_boxes[:, 3] *= scale_y  # y2
            
            # Clip to image boundaries
            scaled_boxes[:, 0] = torch.clamp(scaled_boxes[:, 0], 0, orig_width - 1)
            scaled_boxes[:, 1] = torch.clamp(scaled_boxes[:, 1], 0, orig_height - 1)
            scaled_boxes[:, 2] = torch.clamp(scaled_boxes[:, 2], 0, orig_width - 1)
            scaled_boxes[:, 3] = torch.clamp(scaled_boxes[:, 3], 0, orig_height - 1)
            
            pred['boxes'] = scaled_boxes
        
        # Resize masks
        if 'masks' in pred and len(pred['masks']) > 0:
            masks = pred['masks'].cpu().numpy()
            resized_masks = []
            
            for mask in masks:
                # Get the binary mask (first channel of mask)
                binary_mask = mask[0] > 0.5
                
                # Resize binary mask to original dimensions
                resized_mask = cv2.resize(
                    binary_mask.astype(np.uint8), 
                    (orig_width, orig_height), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Add channel dimension back and convert to tensor
                resized_mask = torch.from_numpy(resized_mask[np.newaxis, :, :].astype(np.float32)).to(pred['masks'].device)
                resized_masks.append(resized_mask)
            
            if resized_masks:
                pred['masks'] = torch.stack(resized_masks)
            else:
                # Empty tensor with appropriate dimensions
                pred['masks'] = torch.zeros((0, 1, orig_height, orig_width), dtype=torch.float32, device=pred['masks'].device)
        
        # Include original dimensions in the prediction
        pred['height'] = orig_height
        pred['width'] = orig_width
        
        resized_predictions.append(pred)
    
    return resized_predictions


def inference(model_path, data_root, output_file, batch_size=1, num_workers=4, 
              model_type='fpn', num_classes=5, device=None, score_threshold=0.5,
              height=1024, width=1024):
    """
    Run inference with a trained model on test data.
    
    Args:
        model_path (str): Path to the trained model checkpoint.
        data_root (str): Root directory for the dataset.
        output_file (str): Path to save predictions.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        model_type (str): Type of model to use ('fpn' or 'fpnv2').
        num_classes (int): Number of classes (including background).
        device (torch.device): Device to run inference on.
        score_threshold (float): Threshold for detection scores.
        height (int): Image height for processing.
        width (int): Image width for processing.
        
    Returns:
        list: Formatted predictions for evaluation.
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create data loader for test data
    test_loader = get_dataloader(
        root_dir=data_root,
        batch_size=batch_size,
        split='test',
        num_workers=num_workers,
        height=height,
        width=width
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
    
    # Load model weights
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        # If the checkpoint has 'model_state_dict' key
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        # If the checkpoint contains the model directly
        model.load_state_dict(state_dict)
    
    # Create model wrapper for inference
    if model_type == 'resnext_fpn':
        model_wrapper = MaskRCNNResNeXt101Wrapper(model, device)
    else:
        model_wrapper = MaskRCNNWrapper(model, device)
    
    # Run inference
    print("Running inference...")
    all_predictions = []

    # count_type_of_label = {}
    
    for images, targets in tqdm(test_loader):
        # Get model predictions
        batch_predictions = model_wrapper(images)
        
        # Post-process predictions
        for i, pred in enumerate(batch_predictions):
            # label = pred['labels'].cpu().numpy()
            # for l in label:
            #     count_type_of_label[l] = count_type_of_label.get(l, 0) + 1

            # Filter by score threshold
            keep_idxs = pred['scores'] > score_threshold
            
            # Create filtered prediction dictionary
            filtered_pred = {
                'boxes': pred['boxes'][keep_idxs],
                'labels': pred['labels'][keep_idxs] - 1,
                'scores': pred['scores'][keep_idxs],
                'masks': pred['masks'][keep_idxs],
                'image_id': targets[i]['image_id'],
                'orig_height': targets[i].get('orig_height'),
                'orig_width': targets[i].get('orig_width')
            }
            
            all_predictions.append(filtered_pred)
    
    # # Print the count of each label
    # print("Count of each label in the predictions:")
    # for label, count in count_type_of_label.items():
    #     print(f"Label {label}: {count} instances")
    # print("Total predictions:", len(all_predictions))

    # Resize predictions to original image dimensions
    print("Resizing predictions to original image dimensions...")
    resized_predictions = resize_predictions_to_original(
        all_predictions, 
        model_input_height=height,
        model_input_width=width
    )
    
    # Format predictions for submission
    formatted_predictions = format_predictions(resized_predictions)
    
    # Save predictions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving predictions to {output_file}")
    save_predictions(formatted_predictions, output_file)
    
    return formatted_predictions
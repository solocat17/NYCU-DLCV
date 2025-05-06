import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils
import json
import os


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)

def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array

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

def format_predictions(predictions):
    """
    Format model predictions for submission.
    
    Args:
        predictions (list): List of prediction dictionaries for each image.
        dataset (MedicalCellDataset): The dataset containing image_id mappings.
        
    Returns:
        list: List of prediction dictionaries in COCO format.
    """
    results = []
    
    for image_preds in predictions:
        image_id = image_preds['image_id']
        boxes = image_preds['boxes'].cpu().numpy()
        scores = image_preds['scores'].cpu().numpy()
        labels = image_preds['labels'].cpu().numpy()
        masks = image_preds['masks'].cpu().numpy()
        
        # Get image dimensions
        # h, w = masks.shape[1:] if masks.size > 0 else (0, 0)
        
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            # Convert mask to RLE format
            binary_mask = mask > 0.5  # Apply threshold
            binary_mask = binary_mask[0]
            rle = encode_mask(binary_mask)
            
            # Format as COCO result
            result = {
                'image_id': image_id,
                'bbox': box.tolist(),  # [x1, y1, x2, y2] -> [x, y, width, height]
                'score': float(score),
                'category_id': int(label) + 1,  # Assuming 1-indexed categories
                'segmentation': rle
            }
            
            results.append(result)
    
    return results


def save_predictions(predictions, output_file):
    """
    Save predictions to a JSON file.
    
    Args:
        predictions (list): List of prediction dictionaries.
        output_file (str): Output file path.
    """
    with open(output_file, 'w') as f:
        json.dump(predictions, f)

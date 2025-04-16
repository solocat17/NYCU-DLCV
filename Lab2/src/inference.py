import torch
import os
import json
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.ops import nms
from PIL import Image

def plot_img_bbox(img, output, save_path=None):
    """Plot image with bounding boxes and save to file"""
    # Convert tensor to numpy image
    if torch.is_tensor(img):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Denormalize
        img = np.clip(img, 0, 1)

    # Resize the image and the boxes to a fixed size: 800x400
    img = Image.fromarray((img * 255).astype(np.uint8))
    original_size = img.size
    # Resize the image by direct upscaling without interpolation
    img = img.resize((800, 400), Image.BILINEAR)
    img = np.array(img) / 255.0
    for box in output['boxes']:
        box[0] = box[0] * (800 / original_size[1])
        box[1] = box[1] * (400 / original_size[0])
        box[2] = box[2] * (800 / original_size[1])
        box[3] = box[3] * (400 / original_size[0])

    # Create figure with dynamic size
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        box = box.detach().cpu().numpy()
        x, y, x2, y2 = box
        width = x2 - x
        height = y2 - y

        # Create rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Convert label to int and score to float
        label = int(label.detach().cpu())
        score = float(score.detach().cpu())

        # Add label text
        plt.text(x, y - 5, f'{label - 1}: {score:.2f}',
                color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close(fig)

def infer(model, test_loader, device, output_dir, confidence_threshold=0.3, nms_threshold=0.1):
    """
    Perform inference on test data
    """
    model.eval()
    all_predictions = []
    output_dir = os.path.join(output_dir, 'visualizations')
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = list(image.to(device) for image in batch['images'])
            image_ids = batch['image_ids']
            original_sizes = batch['original_sizes']
            
            # Get predictions
            predictions = model(images)
            
            for i, pred in enumerate(predictions):
                pred_dict = {
                    'image_id': image_ids[i].item(),
                    'original_size': original_sizes[i],
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                }
                # Confidence filtering
                keep = pred_dict['scores'] > confidence_threshold
                pred_dict['boxes'] = pred_dict['boxes'][keep]
                pred_dict['scores'] = pred_dict['scores'][keep]
                pred_dict['labels'] = pred_dict['labels'][keep]
                
                # Non-Maximum Suppression
                if len(pred_dict['boxes']) > 0:
                    keep = nms(pred_dict['boxes'], pred_dict['scores'], nms_threshold)
                    pred_dict['boxes'] = pred_dict['boxes'][keep]
                    pred_dict['scores'] = pred_dict['scores'][keep]
                    pred_dict['labels'] = pred_dict['labels'][keep]

                    # Remove boxes that are too far apart from each other
                    if len(pred_dict['boxes']) > 1:
                        distances = torch.cdist(pred_dict['boxes'], pred_dict['boxes'])
                        # normalize the distances
                        distances = distances / torch.max(distances)
                        # remove boxes that are too far apart from all others
                        keep = distances.mean(dim=1) <= 0.6
                        pred_dict['boxes'] = pred_dict['boxes'][keep]
                        pred_dict['scores'] = pred_dict['scores'][keep]
                        pred_dict['labels'] = pred_dict['labels'][keep]
                else:
                    pred_dict['boxes'] = torch.empty((0, 4))
                    pred_dict['scores'] = torch.empty((0,))
                    pred_dict['labels'] = torch.empty((0,), dtype=torch.int64)

                # Resize the image and boxes to original size
                from torchvision.transforms import functional as F
                # new_width, new_height = 400, 400
                # scale_width = new_width / image.width
                # scale_height = new_height / image.height
                # image = F.resize(image, [new_height, new_width])
                scale_width = original_sizes[i][0] / 400
                scale_height = original_sizes[i][1] / 400
                images[i] = F.resize(images[i], original_sizes[i])
                # if len(boxes) > 0:
                # boxes[:, [0, 2]] *= scale_width  # Scale x coordinates
                # boxes[:, [1, 3]] *= scale_height  # Scale y coordinates
                if len(pred_dict['boxes']) > 0:
                    pred_dict['boxes'][:, [0, 2]] *= scale_width
                    pred_dict['boxes'][:, [1, 3]] *= scale_height
                    # Clipping boxes to image size
                    pred_dict['boxes'] = torch.clamp(pred_dict['boxes'], min=0)
                    pred_dict['boxes'][:, 0] = torch.clamp(pred_dict['boxes'][:, 0], max=original_sizes[i][0])
                    pred_dict['boxes'][:, 1] = torch.clamp(pred_dict['boxes'][:, 1], max=original_sizes[i][1])
                    pred_dict['boxes'][:, 2] = torch.clamp(pred_dict['boxes'][:, 2], max=original_sizes[i][0])
                    pred_dict['boxes'][:, 3] = torch.clamp(pred_dict['boxes'][:, 3], max=original_sizes[i][1])    
                
                all_predictions.append(pred_dict)

            # Save first 50 images with predictions
            if image_ids[i].item() < 50:
                img_path = os.path.join(output_dir, f"{image_ids[i].item()}.png")
                # print(f"Image size: {images[i].shape}")
                plot_img_bbox(images[i], pred_dict, save_path=img_path)

    return all_predictions

def generate_task1_submission(predictions, output_file):
    """
    Generate submission file for Task 1 (digit detection)
    """
    result = []
    
    for pred in predictions:
        image_id = pred['image_id']
        boxes = pred['boxes'].numpy()
        scores = pred['scores'].numpy()
        labels = pred['labels'].numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            #if score > 0.5:
            # COCO format expects [x_min, y_min, width, height]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Output the raw label (1-10) as category_id
            result.append({
                'image_id': int(image_id),
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'score': float(score),
                'category_id': int(label)  # Keep as-is (1-10)
            })

    with open(output_file, 'w') as f:
        json.dump(result, f)

def recognize_number(boxes, labels, scores, threshold=0.5):
    valid_indices = scores > threshold
    if not valid_indices.any():
        return -1
    
    filtered_boxes = boxes[valid_indices]
    filtered_labels = labels[valid_indices]
    
    # Convert tensor to numpy if needed
    if isinstance(filtered_boxes, torch.Tensor):
        filtered_boxes = filtered_boxes.cpu().numpy()
    if isinstance(filtered_labels, torch.Tensor):
        filtered_labels = filtered_labels.cpu().numpy()
    
    # Sort boxes from left to right
    sorted_indices = filtered_boxes[:, 0].argsort()
    sorted_labels = filtered_labels[sorted_indices]
    
    # Convert model outputs (1-10) to actual digits (0-9)
    digits = [str(int(l) - 1) for l in sorted_labels]
    
    if not digits:
        return -1
    
    try:
        number = int(''.join(digits))
        return number
    except ValueError:
        return -1

def generate_task2_submission(predictions, output_file):
    """
    Generate submission file for Task 2 (number recognition)
    """
    results = {}
    
    for pred in predictions:
        image_id = pred['image_id']
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']
        
        number = recognize_number(boxes, labels, scores)
        results[image_id] = number
    
    # Write to CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'pred_label'])
        for image_id, number in sorted(results.items()):
            writer.writerow([image_id, number])
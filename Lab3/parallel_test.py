import os
import torch
import torch.multiprocessing as mp
import argparse
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np
import cv2
from functools import partial
import datetime
import zipfile

from src.model import get_maskrcnn_model, get_maskrcnn_model_fpnv2, get_maskrcnn_resnext101_fpn, get_maskrcnn_resnext101_unet
from src.model import MaskRCNNWrapper, MaskRCNNResNeXt101Wrapper
from src.dataloader import get_test_transforms
from src.utils import format_predictions, save_predictions
from src.inference import resize_predictions_to_original


def setup_model(model_path, model_type, num_classes, device):
    """
    Setup the model on the specified device.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model to use
        num_classes (int): Number of classes (including background)
        device (torch.device): Device to load the model onto
        
    Returns:
        MaskRCNNWrapper: Wrapped model ready for inference
    """
    print(f"Setting up model of type {model_type} on {device}")
    
    # Create model based on the model type
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
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    # Create model wrapper for inference
    if model_type == 'resnext_fpn':
        model_wrapper = MaskRCNNResNeXt101Wrapper(model, device)
    else:
        model_wrapper = MaskRCNNWrapper(model, device)
    
    return model_wrapper


def process_batch(batch_files, model_wrapper, transform, score_threshold, 
                  height, width, device, image_metadata=None):
    """
    Process a batch of images with the given model.
    
    Args:
        batch_files (list): List of image file paths
        model_wrapper (MaskRCNNWrapper): Wrapped model for inference
        transform: Image transformations to apply
        score_threshold (float): Score threshold for predictions
        height (int): Model input height
        width (int): Model input width
        device (torch.device): Device to run inference on
        image_metadata (dict): Metadata for each image
        
    Returns:
        list: List of prediction dictionaries
    """
    images = []
    targets = []
    
    # Load and preprocess images
    for file_path in batch_files:
        # Read image
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get filename and metadata
        filename = Path(file_path).name
        orig_height = None
        orig_width = None
        image_id = None
        
        if image_metadata and filename in image_metadata:
            metadata = image_metadata[filename]
            orig_height = metadata['height']
            orig_width = metadata['width']
            image_id = metadata['id']
        else:
            # If not in metadata, use the dimensions from the image itself
            orig_height, orig_width = image.shape[:2]
            image_id = filename
        
        # Apply transforms
        transformed = transform(image=image)
        processed_image = transformed['image'].to(device)
        
        images.append(processed_image)
        targets.append({
            'image_id': image_id,
            'orig_height': orig_height,
            'orig_width': orig_width
        })
    
    # Run inference
    with torch.no_grad():
        batch_predictions = model_wrapper(images)
    
    # Post-process predictions
    all_predictions = []
    for i, pred in enumerate(batch_predictions):
        # Filter by score threshold
        keep_idxs = pred['scores'] > score_threshold
        
        # Create filtered prediction dictionary
        filtered_pred = {
            'boxes': pred['boxes'][keep_idxs],
            'labels': pred['labels'][keep_idxs] - 1,  # Convert to 0-indexed
            'scores': pred['scores'][keep_idxs],
            'masks': pred['masks'][keep_idxs],
            'image_id': targets[i]['image_id'],
            'orig_height': targets[i]['orig_height'],
            'orig_width': targets[i]['orig_width']
        }
        
        all_predictions.append(filtered_pred)
    
    return all_predictions


def load_image_metadata(data_root):
    """
    Load image metadata from the test image mapping file.
    
    Args:
        data_root (str): Root directory for the dataset
        
    Returns:
        dict: Dictionary mapping filenames to metadata
    """
    metadata = {}
    mapping_path = Path(data_root) / 'dataset' / 'test_image_name_to_ids.json'
    
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
            for item in mapping_data:
                metadata[item['file_name']] = {
                    'id': item['id'],
                    'height': item['height'],
                    'width': item['width']
                }
    
    return metadata


def worker_process(gpu_id, num_gpus, model_path, model_type, num_classes, 
                  batch_files, score_threshold, height, width, 
                  result_queue, image_metadata=None):
    """
    Worker process function for parallel inference.
    
    Args:
        gpu_id (int): ID of the GPU for this worker
        num_gpus (int): Total number of GPUs
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model
        num_classes (int): Number of classes
        batch_files (list): List of image file paths assigned to this worker
        score_threshold (float): Score threshold for predictions
        height (int): Model input height
        width (int): Model input width
        result_queue (mp.Queue): Queue to store results
        image_metadata (dict): Metadata for each image
    """
    # Set the device for this worker
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Set up model
    model_wrapper = setup_model(model_path, model_type, num_classes, device)
    
    # Set up transforms
    transform = get_test_transforms(height, width)
    
    # Process batches
    predictions = []
    
    # Split batch_files into smaller batches
    batch_size = 4  # Adjust based on GPU memory
    
    for i in range(0, len(batch_files), batch_size):
        mini_batch = batch_files[i:i+batch_size]
        batch_predictions = process_batch(
            mini_batch, model_wrapper, transform, score_threshold,
            height, width, device, image_metadata
        )
        predictions.extend(batch_predictions)
    
    # Resize predictions to original image dimensions
    resized_predictions = resize_predictions_to_original(
        predictions, 
        model_input_height=height,
        model_input_width=width
    )
    
    # Put the results in the queue
    result_queue.put(resized_predictions)

    print(f"Worker {gpu_id} finished processing {len(batch_files)} files.")


def parallel_inference(model_path, data_root, zip_name, output_file, batch_size=1, 
                      model_type='fpn', num_classes=5, score_threshold=0.5,
                      height=1024, width=1024, num_workers=None):
    """
    Run inference in parallel across multiple GPUs.
    
    Args:
        model_path (str): Path to the model checkpoint
        data_root (str): Root directory for the dataset
        zip_name (str): Path to save the zip file
        output_file (str): Path to save predictions
        batch_size (int): Batch size for each device
        model_type (str): Type of model to use
        num_classes (int): Number of classes (including background)
        score_threshold (float): Threshold for detection scores
        height (int): Image height for processing
        width (int): Image width for processing
        num_workers (int): Number of worker processes (defaults to number of GPUs)
        
    Returns:
        list: Formatted predictions for evaluation
    """
    # Check for available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        device = torch.device('cpu')
        
        # Fall back to single process inference
        from src.inference import inference
        return inference(
            model_path=model_path,
            data_root=data_root,
            output_file=output_file,
            batch_size=batch_size,
            num_workers=4,
            model_type=model_type,
            num_classes=num_classes,
            device=device,
            score_threshold=score_threshold,
            height=height,
            width=width
        )
    
    # Set number of workers
    if num_workers is None:
        num_workers = num_gpus
    
    print(f"Running parallel inference with {num_workers} workers on {num_gpus} GPUs")
    
    # Get all test image files
    test_dir = Path(data_root) / 'dataset' / 'test'
    test_files = list(test_dir.glob('*.tif'))
    print(f"Found {len(test_files)} test images")
    
    # Load image metadata
    image_metadata = load_image_metadata(data_root)
    
    # Split the files among workers
    files_per_worker = [[] for _ in range(num_workers)]
    for i, file_path in enumerate(test_files):
        worker_idx = i % num_workers
        files_per_worker[worker_idx].append(file_path)
    
    # Create a queue for results
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for gpu_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id % num_gpus,  # GPU ID
                num_gpus,
                model_path,
                model_type,
                num_classes,
                files_per_worker[gpu_id],
                score_threshold,
                height,
                width,
                result_queue,
                image_metadata
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results
    all_predictions = []
    for _ in range(num_workers):
        worker_predictions = result_queue.get()
        all_predictions.extend(worker_predictions)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Format predictions for submission
    formatted_predictions = format_predictions(all_predictions)
    
    # Create output directory if it doesn't exist
    model_dir = Path(model_path).parent
    output_file = model_dir / output_file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    print(f"Saving predictions to {output_file}")
    save_predictions(formatted_predictions, output_file)
    

    # Create a timestamp for the zip file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a zip file of the results under the same directory
    zip_dir = output_path.parent
    zip_filename = zip_dir / f"{zip_name}_{timestamp}.zip"
    # zip_filename = zip_dir / f"{output_path.stem}_{timestamp}_solution.zip"
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add the output file to the zip
        zipf.write(output_file, arcname=output_path.name)

    return formatted_predictions


if __name__ == "__main__":
    # Enable multiprocessing for PyTorch
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Run parallel inference on multiple GPUs")
    parser.add_argument("--model_path", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data_root", default='.', help="Root directory for the dataset")
    parser.add_argument("--zip_name", required=True, help="Path to save the zip file")
    parser.add_argument("--output_file", default='test-results.json', help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for each GPU")
    parser.add_argument("--model_type", choices=['fpn', 'fpnv2', 'resnext_fpn', 'resnext_unet'], default='fpn', help="Type of model to use")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (including background)")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Threshold for detection scores")
    parser.add_argument("--height", type=int, default=1024, help="Image height for processing")
    parser.add_argument("--width", type=int, default=1024, help="Image width for processing")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (defaults to number of GPUs)")
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    
    # Run parallel inference
    predictions = parallel_inference(
        model_path=args.model_path,
        data_root=args.data_root,
        zip_name=args.zip_name,
        output_file=args.output_file,
        batch_size=args.batch_size,
        model_type=args.model_type,
        num_classes=args.num_classes,
        score_threshold=args.score_threshold,
        height=args.height,
        width=args.width,
        num_workers=args.num_workers
    )
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Parallel inference completed in {elapsed_time:.2f} seconds")
    print(f"Number of predictions: {len(predictions)}")
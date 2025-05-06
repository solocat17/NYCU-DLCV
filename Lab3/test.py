import os
import torch
import argparse
from pathlib import Path
import zipfile
import datetime

from src.inference import inference


def test_model(model_path, data_root, zip_name, output_file, batch_size=1, 
               num_workers=4, model_type='fpn', num_classes=5, 
               score_threshold=0.5, height=1024, width=1024):
    """
    Test a trained Mask R-CNN model and save predictions.
    
    Args:
        model_path (str): Path to the trained model checkpoint.
        data_root (str): Root directory for the dataset.
        zip_name (str): Name of the zip file to save predictions.
        output_file (str): Path to save predictions.
        batch_size (int): Batch size for inference.
        num_workers (int): Number of workers for data loading.
        model_type (str): Type of model to use ('fpn' or 'fpnv2').
        num_classes (int): Number of classes (including background).
        score_threshold (float): Threshold for detection scores.
        height (int): Image height for processing.
        width (int): Image width for processing.
        
    Returns:
        list: Formatted predictions for evaluation.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_dir = Path(model_path).parent
    output_file = model_dir / output_file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    predictions = inference(
        model_path=model_path,
        data_root=data_root,
        output_file=output_file,
        batch_size=batch_size,
        num_workers=num_workers,
        model_type=model_type,
        num_classes=num_classes,
        device=device,
        score_threshold=score_threshold,
        height=height,
        width=width
    )

    # Create a timestamp for the zip file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a zip file of the results under the same directory
    zip_dir = output_path.parent
    zip_filename = zip_dir / f"{zip_name}_{timestamp}.zip"
    # zip_filename = zip_dir / f"{output_path.stem}_{timestamp}_solution.zip"
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Add the output file to the zip
        zipf.write(output_file, arcname=output_path.name)
    
    print(f"Results zipped into {zip_filename}")
    print(f"Testing completed. Predictions saved to {output_file}")
    print(f"Number of predictions: {len(predictions)}")
    
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained Mask R-CNN model")
    parser.add_argument("--model_path", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data_root", default='.', help="Root directory for the dataset")
    parser.add_argument("--zip_name", required=True, help="Path to save the zip file")
    parser.add_argument("--output_file", default='test-results.json', help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--model_type", choices=['fpn', 'fpnv2', 'resnext_fpn', 'resnext_unet'], default='fpn', help="Type of model to use")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes (including background)")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Threshold for detection scores")
    parser.add_argument("--height", type=int, default=1024, help="Image height for processing")
    parser.add_argument("--width", type=int, default=1024, help="Image width for processing")
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model_path,
        data_root=args.data_root,
        zip_name=args.zip_name,
        output_file=args.output_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model_type,
        num_classes=args.num_classes,
        score_threshold=args.score_threshold,
        height=args.height,
        width=args.width
    )
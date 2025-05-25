import os
import numpy as np
import zipfile
import argparse
import datetime
from tqdm import tqdm
import tempfile
import shutil

def load_predictions_from_zip(zip_path):
    """Load predictions from a zip file containing .npz file"""
    predictions = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Find the .npz file in the zip
        npz_files = [f for f in zipf.namelist() if f.endswith('.npz')]
        
        if len(npz_files) == 0:
            raise ValueError(f"No .npz file found in {zip_path}")
        elif len(npz_files) > 1:
            print(f"Warning: Multiple .npz files found in {zip_path}, using the first one: {npz_files[0]}")
        
        npz_file = npz_files[0]
        
        # Extract the .npz file to a temporary location
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as temp_file:
            temp_file.write(zipf.read(npz_file))
            temp_path = temp_file.name
        
        # Load the .npz file
        data = np.load(temp_path)
        for key in data.files:
            predictions[key] = data[key]
        
        # Clean up temporary file
        os.unlink(temp_path)
    
    return predictions

def load_predictions_from_npz(npz_path):
    """Load predictions directly from .npz file"""
    predictions = {}
    data = np.load(npz_path)
    for key in data.files:
        predictions[key] = data[key]
    return predictions

def ensemble_predictions(prediction_files, weights=None):
    """
    Ensemble multiple prediction files by averaging
    
    Args:
        prediction_files: List of file paths (can be .zip or .npz files)
        weights: Optional list of weights for each model (default: equal weights)
    
    Returns:
        Dictionary of ensembled predictions
    """
    
    if weights is None:
        weights = [1.0 / len(prediction_files)] * len(prediction_files)
    elif len(weights) != len(prediction_files):
        raise ValueError("Number of weights must match number of prediction files")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"Ensembling {len(prediction_files)} models with weights: {weights}")
    
    all_predictions = []
    
    # Load all predictions
    for i, file_path in enumerate(prediction_files):
        print(f"Loading predictions from {file_path}...")
        
        if file_path.endswith('.zip'):
            predictions = load_predictions_from_zip(file_path)
        elif file_path.endswith('.npz'):
            predictions = load_predictions_from_npz(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Only .zip and .npz files are supported.")
        
        all_predictions.append(predictions)
        print(f"  Loaded {len(predictions)} images")
    
    # Check that all models have predictions for the same images
    all_keys = set(all_predictions[0].keys())
    for i, predictions in enumerate(all_predictions[1:], 1):
        current_keys = set(predictions.keys())
        if current_keys != all_keys:
            missing_in_current = all_keys - current_keys
            extra_in_current = current_keys - all_keys
            
            if missing_in_current:
                print(f"Warning: Model {i} is missing predictions for: {missing_in_current}")
            if extra_in_current:
                print(f"Warning: Model {i} has extra predictions for: {extra_in_current}")
            
            # Use intersection of all keys
            all_keys = all_keys.intersection(current_keys)
    
    print(f"Ensembling predictions for {len(all_keys)} common images")
    
    # Ensemble predictions
    ensembled_predictions = {}
    
    for image_key in tqdm(all_keys, desc="Ensembling images"):
        # Stack all predictions for this image
        image_predictions = []
        for predictions in all_predictions:
            image_predictions.append(predictions[image_key].astype(np.float32))
        
        # Convert to numpy array: (num_models, C, H, W)
        image_predictions = np.stack(image_predictions, axis=0)
        
        # Weighted average
        ensembled_image = np.average(image_predictions, axis=0, weights=weights)
        
        # Convert back to uint8
        ensembled_image = np.clip(ensembled_image, 0, 255).astype(np.uint8)
        
        ensembled_predictions[image_key] = ensembled_image
    
    return ensembled_predictions

def save_ensembled_predictions(ensembled_predictions, output_dir, output_name=None):
    """Save ensembled predictions to .npz and .zip files"""
    
    if output_name is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"ensembled_predictions_{timestamp}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to .npz file
    npz_path = os.path.join(output_dir, "pred.npz")
    print(f"Saving {len(ensembled_predictions)} ensembled images to {npz_path}")
    np.savez(npz_path, **ensembled_predictions)
    
    # Create zip file
    zip_path = os.path.join(output_dir, f"{output_name}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(npz_path, os.path.basename(npz_path))
    
    print(f"Created zip file: {zip_path}")
    
    return zip_path

def main(args):
    # Validate input files
    for file_path in args.prediction_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
    
    # Parse weights if provided
    weights = None
    if args.weights:
        weights = [float(w) for w in args.weights.split(',')]
        if len(weights) != len(args.prediction_files):
            raise ValueError("Number of weights must match number of prediction files")
    
    # Ensemble predictions
    ensembled_predictions = ensemble_predictions(args.prediction_files, weights)
    
    # Save results
    output_path = save_ensembled_predictions(
        ensembled_predictions, 
        args.output_dir, 
        args.output_name
    )
    
    print(f"Ensemble complete! Results saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble multiple model predictions')
    parser.add_argument('prediction_files', nargs='+', 
                        help='Paths to prediction files (.zip or .npz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save ensembled results')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Name for output files (without extension)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated weights for each model (e.g., "0.4,0.6")')
    
    args = parser.parse_args()
    main(args)
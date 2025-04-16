import argparse
import os
import torch
import yaml

from src.dataloader import get_dataloaders
from src.model import DigitDetector
from src.inference import infer, generate_task1_submission, generate_task2_submission

def parse_args():
    parser = argparse.ArgumentParser(description="Test Digit Recognition Model")
    parser.add_argument('--config', type=str, default='configs/default.yml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to data directory')
    parser.add_argument('--exp_name', type=str, default='exp1', help='Experiment name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    result_dir = os.path.join('results', args.exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    _, _, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=config['testing']['batch_size']
    )
    
    # Create model
    model = DigitDetector(num_classes=11)  # 10 digits (0-9) + background
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    
    # Perform inference
    predictions = infer(model, test_loader, device, result_dir)
    
    # Generate submission files
    generate_task1_submission(predictions, os.path.join(result_dir, 'pred.json'))
    generate_task2_submission(predictions, os.path.join(result_dir, 'pred.csv'))
    print("Testing completed!")
    print(f"Submission files saved to {result_dir}/pred.json and {result_dir}/pred.csv")
    
    import zipfile
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # Create a zip file of the results with name '%{timestamp}_solution.zip'
    zip_filename = os.path.join(result_dir, f'{timestamp}_solution.zip')
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(os.path.join(result_dir, 'pred.json'), arcname='pred.json')
        zipf.write(os.path.join(result_dir, 'pred.csv'), arcname='pred.csv')
    
    print(f"Results zipped to {zip_filename}")

if __name__ == "__main__":
    main()
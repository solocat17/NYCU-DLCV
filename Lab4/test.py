import os
import torch
import zipfile
import datetime
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from src.promptir_model import PromptIR
from src.dataloader import get_dataloaders

def tensor_to_numpy(tensor):
    """Convert a torch tensor to a numpy array in the required format (3, H, W) for submission"""
    # Move to CPU and detach from computation graph
    img = tensor.detach().cpu()
    
    # Unnormalize the image
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # Clamp to valid range [0, 1]
    img = img.clamp(0, 1)
    
    # Convert to numpy and scale to 0-255 range, then convert to uint8
    img_np = (img.numpy() * 255).astype(np.uint8)
    
    return img_np

def main(args):
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = PromptIR(decoder=True)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # PyTorch Lightning checkpoint
        state_dict = checkpoint['state_dict']
        # Remove the 'net.' prefix if it exists
        state_dict = {k.replace('net.', ''): v for k, v in state_dict.items() if 'net.' in k}
        model.load_state_dict(state_dict)
    else:
        # Regular PyTorch checkpoint
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")

    # Get test dataloader
    train_loader, val_loader, test_loader = get_dataloaders(args.data_root, batch_size=args.batch_size)
    
    print(f"Testing with {len(test_loader.dataset)} images...")
    
    # Dictionary to store restored images
    restored_images = {}
    
    # Metrics tracking
    total_psnr = 0.0
    rain_psnr_sum = 0.0
    snow_psnr_sum = 0.0
    rain_count = 0
    snow_count = 0
    
    # Process test images
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            degraded_imgs = batch['degraded'].to(device)
            clean_imgs = batch['clean'].to(device) if 'clean' in batch else None
            degradation_types = batch['degradation_type'] if 'degradation_type' in batch else None
            filenames = batch['degraded_path']
            
            # Forward pass
            restored = model(degraded_imgs)
            
            # Calculate PSNR if ground truth is available
            if clean_imgs is not None:
                # Unnormalize images for PSNR calculation
                def unnormalize(img):
                    return img * torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1) + \
                           torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                
                restored_unorm = unnormalize(restored)
                clean_unorm = unnormalize(clean_imgs)
                
                # Calculate PSNR
                batch_psnr = psnr(restored_unorm, clean_unorm, data_range=1.0).item()
                total_psnr += batch_psnr * len(degraded_imgs)
                
                # Calculate PSNR separately for rain and snow
                if degradation_types is not None:
                    for i, deg_type in enumerate(degradation_types):
                        if deg_type.item() == 0:  # Rain
                            rain_psnr = psnr(restored_unorm[i:i+1], clean_unorm[i:i+1], data_range=1.0).item()
                            rain_psnr_sum += rain_psnr
                            rain_count += 1
                        else:  # Snow
                            snow_psnr = psnr(restored_unorm[i:i+1], clean_unorm[i:i+1], data_range=1.0).item()
                            snow_psnr_sum += snow_psnr
                            snow_count += 1
            
            # Convert to numpy and store
            for i, filename in enumerate(filenames):
                restored_img = tensor_to_numpy(restored[i])
                restored_images[filename] = restored_img
    
    # Print PSNR results if ground truth was available
    if 'clean' in batch:
        avg_psnr = total_psnr / len(test_loader.dataset)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        
        if rain_count > 0:
            print(f"Average PSNR for Rain images: {rain_psnr_sum / rain_count:.2f} dB")
        if snow_count > 0:
            print(f"Average PSNR for Snow images: {snow_psnr_sum / snow_count:.2f} dB")
    
    args.output = os.path.join(args.result_dir, args.output)

    # Save to npz file
    print(f"Saving {len(restored_images)} images to {args.output}")
    np.savez(args.output, **restored_images)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_dir = os.path.dirname(args.output)
    filename = os.path.basename(args.checkpoint)[:-5]
    zip_name = filename.split('-')[1] + '-' + filename.split('-')[2]
    
    zip_filename = os.path.join(zip_dir, f"{zip_name}_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write(args.output, os.path.basename(args.output))

    print(f"Saved results to {zip_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PromptIR model')
    parser.add_argument('--data_root', type=str, default='./dataset',
                        help='Path to the dataset')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--output', type=str, default='pred.npz',
                        help='Output file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    
    args = parser.parse_args()
    main(args)
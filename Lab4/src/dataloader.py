import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class DualDegradationDataset(Dataset):
    """Dataset for both rain and snow degraded images"""
    
    def __init__(self, root_dir, phase='train', crop_size=128, flip_augment=True, rotation_augment=True):
        """
        Args:
            root_dir (str): Root directory of the dataset
            phase (str): 'train', 'val', or 'test'
            crop_size (int): Size of random crops during training
            flip_augment (bool): Whether to use random flips for augmentation
            rotation_augment (bool): Whether to use random rotations for augmentation
        """
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        self.flip_augment = flip_augment
        self.rotation_augment = rotation_augment
        
        # Paths setup
        if phase in ['train', 'val']:
            self.clean_dir = os.path.join(root_dir, 'train', 'clean')
            self.degraded_dir = os.path.join(root_dir, 'train', 'degraded')
            
            # Get all clean images
            rain_clean_paths = sorted(glob.glob(os.path.join(self.clean_dir, 'rain_clean-*.png')))
            snow_clean_paths = sorted(glob.glob(os.path.join(self.clean_dir, 'snow_clean-*.png')))
            
            # Get all degraded images
            rain_degraded_paths = sorted(glob.glob(os.path.join(self.degraded_dir, 'rain-*.png')))
            snow_degraded_paths = sorted(glob.glob(os.path.join(self.degraded_dir, 'snow-*.png')))
            
            # Create pairs of degraded and clean images
            self.degraded_paths = rain_degraded_paths + snow_degraded_paths
            self.clean_paths = rain_clean_paths + snow_clean_paths
            
            # Create degradation type flags (0 for rain, 1 for snow)
            self.degradation_types = [0] * len(rain_degraded_paths) + [1] * len(snow_degraded_paths)
            
        elif phase == 'test':
            self.degraded_dir = os.path.join(root_dir, 'test', 'degraded')
            self.degraded_paths = sorted(glob.glob(os.path.join(self.degraded_dir, '*.png')))
            self.clean_paths = None  # No clean images for test set
            self.degradation_types = None  # No degradation type labels for test set
        
        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        print(f"Creating dataset for {phase} phase, dataset root dir: {root_dir}")
        if phase in ['train', 'val']:
            print(f"  Degradation types: {self.degradation_types.count(0)} rain, {self.degradation_types.count(1)} snow")
            print(f"  Found {len(self.degraded_paths)} degraded images from {self.degraded_dir}")
            print(f"  Found {len(self.clean_paths)} clean images from {self.clean_dir}")
        else:
            print(f"  Found {len(self.degraded_paths)} test images from {self.degraded_dir}")
    
    def __len__(self):
        return len(self.degraded_paths)
    
    def __getitem__(self, idx):
        # Load degraded image
        degraded_img_path = self.degraded_paths[idx]
        degraded_img = Image.open(degraded_img_path).convert('RGB')
        
        # For training and validation phases, load the corresponding clean image
        if self.phase in ['train', 'val']:
            clean_img_path = self.clean_paths[idx]
            clean_img = Image.open(clean_img_path).convert('RGB')
            degradation_type = self.degradation_types[idx]
            
            # Apply the same transformations to both images
            if self.crop_size and self.phase == 'train':
                # Random crop for training only
                i, j, h, w = transforms.RandomCrop.get_params(
                    clean_img, output_size=(self.crop_size, self.crop_size))
                clean_img = TF.crop(clean_img, i, j, h, w)
                degraded_img = TF.crop(degraded_img, i, j, h, w)
            elif self.crop_size and self.phase == 'val':
                # Center crop for validation 
                clean_img = TF.center_crop(clean_img, self.crop_size)
                degraded_img = TF.center_crop(degraded_img, self.crop_size)
            
            # Apply augmentations only during training
            if self.phase == 'train':
                # Random horizontal flipping
                if self.flip_augment and random.random() > 0.5:
                    clean_img = TF.hflip(clean_img)
                    degraded_img = TF.hflip(degraded_img)
                
                # Random vertical flipping
                if self.flip_augment and random.random() > 0.5:
                    clean_img = TF.vflip(clean_img)
                    degraded_img = TF.vflip(degraded_img)
                
                # Random rotation
                if self.rotation_augment and random.random() > 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    clean_img = TF.rotate(clean_img, angle)
                    degraded_img = TF.rotate(degraded_img, angle)
            
            # Convert to tensor and normalize
            clean_img = self.normalize(self.to_tensor(clean_img))
            degraded_img = self.normalize(self.to_tensor(degraded_img))
            
            # Return a dictionary with all information
            return {
                'degraded': degraded_img,
                'clean': clean_img,
                'degradation_type': degradation_type,
                'degraded_path': os.path.basename(degraded_img_path)
            }
        
        # For test phase, only return the degraded image
        else:
            # Convert to tensor and normalize
            degraded_img = self.normalize(self.to_tensor(degraded_img))
            
            # Return a dictionary with test information
            return {
                'degraded': degraded_img,
                'degraded_path': os.path.basename(degraded_img_path)
            }


class TrainValDatasetWrapper(Dataset):
    """A wrapper dataset for handling dataset subset indices"""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_dataloaders(data_root, batch_size=4, num_workers=4, crop_size=128, val_split=0.2, seed=42):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_root (str): Path to dataset root directory
        batch_size (int): Batch size
        num_workers (int): Number of workers for dataloader
        crop_size (int): Size of random crops during training
        val_split (float): Proportion of training data to use for validation (0.0 to 1.0)
        seed (int): Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create full training dataset
    full_train_dataset = DualDegradationDataset(
        root_dir=data_root,
        phase='train',
        crop_size=crop_size,
        flip_augment=True,
        rotation_augment=True
    )
    
    # Calculate sizes for train and validation split
    dataset_size = len(full_train_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Create train/validation splits
    indices = list(range(dataset_size))
    # Shuffle indices to ensure random split
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create wrapped datasets for train and validation
    train_dataset = TrainValDatasetWrapper(full_train_dataset, train_indices)
    val_dataset = TrainValDatasetWrapper(full_train_dataset, val_indices)
    
    # Create test dataset
    test_dataset = DualDegradationDataset(
        root_dir=data_root,
        phase='test',
        crop_size=None,  # No random cropping for test set
        flip_augment=False,
        rotation_augment=False
    )
    
    print(f"Creating dataloaders with batch size {batch_size}, crop size {crop_size}, and validation split {val_split}")
    print(f"Train/Val split: {train_size}/{val_size} images")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # Process one image at a time during testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataloaders created:")
    print(f"  Train loader size: {len(train_loader)} batches")
    print(f"  Validation loader size: {len(val_loader)} batches")
    print(f"  Test loader size: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":
    # Example of how to use the dataloader
    data_root = "/project3/yen/DLCV/Lab4/dataset"
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root, 
        batch_size=8,
        val_split=0.2,  # 20% of training data will be used for validation
        seed=42  # For reproducible splits
    )
    
    # Print dataset sizes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Visualize a batch (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        for batch in train_loader:
            degraded_imgs = batch['degraded']
            clean_imgs = batch['clean']
            degradation_types = batch['degradation_type']
            
            # Convert from normalized tensor to display format
            def tensor_to_img(tensor):
                img = tensor.clone().detach().cpu()  # No unnormalization
                # Unnormalize mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img.clamp(0, 1)  # Clamp to [0, 1]
                img = img.numpy().transpose(1, 2, 0)  # Change from CxHxW to HxWxC
                img = np.clip(img, 0, 1)
                return img
            
            # Display first 4 samples in batch
            plt.figure(figsize=(12, 6))
            for i in range(min(4, len(degraded_imgs))):
                degraded = tensor_to_img(degraded_imgs[i])
                clean = tensor_to_img(clean_imgs[i])
                deg_type = "Rain" if degradation_types[i] == 0 else "Snow"
                
                plt.subplot(2, 4, i+1)
                plt.imshow(degraded)
                plt.title(f"Degraded ({deg_type})")
                plt.axis('off')
                
                plt.subplot(2, 4, i+5)
                plt.imshow(clean)
                plt.title("Clean")
                plt.axis('off')
            
            plt.tight_layout()
            # Save the figure
            plt.savefig("sample_batch.png")
            break
            
    except ImportError:
        print("Matplotlib not available for visualization")
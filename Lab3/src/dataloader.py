import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import cv2
import skimage.io as sio
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils import read_maskfile


class MedicalCellDataset(Dataset):
    """
    Dataset class for medical cell instance segmentation
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory for the dataset.
            split (str): 'train' or 'test' split.
            transform (callable, optional): Optional transform to be applied to samples.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Class names and mapping
        self.class_names = ['background', 'class1', 'class2', 'class3', 'class4']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load image id mapping for test set (containing original dimensions)
        self.image_metadata = {}
        if split == 'test':
            mapping_path = self.root_dir / 'dataset' / 'test_image_name_to_ids.json'
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    # Create metadata including dimensions
                    for item in mapping_data:
                        self.image_metadata[item['file_name']] = {
                            'id': item['id'],
                            'height': item['height'],
                            'width': item['width']
                        }
        
        # Get all samples
        self.samples = self._get_samples()
        
        # Define default transforms if none provided
        if self.transform is None and split == 'train':
            self.transform = get_train_transforms()
        elif self.transform is None and split == 'test':
            self.transform = get_test_transforms()
        
    def _get_samples(self):
        """
        Get all samples from the dataset directory.
        
        Returns:
            List of sample paths for train or test split.
        """
        split_dir = self.root_dir / 'dataset' / self.split
        
        if self.split == 'train':
            # For training data, get all image directories
            return [d for d in split_dir.iterdir() if d.is_dir()]
        else:
            # For test data, get all image paths
            image_paths = list(split_dir.glob('*.tif'))
            return image_paths
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            A dictionary containing the image and masks.
        """
        sample_path = self.samples[idx]
        
        if self.split == 'train':
            # For training data, load image and all available masks
            image_path = sample_path / 'image.tif'
            # Read image as RGB (OpenCV reads as BGR)
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Store original dimensions
            orig_height, orig_width = image.shape[:2]
            
            # Check which class masks are available
            masks = []
            mask_labels = []
            
            for class_name in self.class_names:
                mask_path = sample_path / f"{class_name}.tif"
                if mask_path.exists():
                    mask = read_maskfile(str(mask_path))
                    
                    # Handle multi-instance masks
                    # Assuming each unique non-zero value in the mask represents an instance
                    unique_ids = np.unique(mask)
                    unique_ids = unique_ids[unique_ids > 0]  # Skip background (0)
                    
                    for instance_id in unique_ids:
                        instance_mask = (mask == instance_id).astype(np.uint8)
                        masks.append(instance_mask)
                        mask_labels.append(self.class_to_idx[class_name])
            
            # Handle case with no masks
            if len(masks) == 0:
                masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
                mask_labels = []
            else:
                masks = np.stack(masks, axis=0)
            
            # Apply augmentations and transforms
            if self.transform:
                # For albumentations, we need to prepare the masks in the right format
                # Try up to 10 times to get valid augmentations if any masks exist
                max_attempts = 10 if len(mask_labels) > 0 else 1
                
                for attempt in range(max_attempts):
                    transformed = apply_transforms(self.transform, image, masks)
                    valid_masks = []
                    valid_labels = []
                    
                    # Collect valid transformed masks
                    if len(masks) > 0:
                        for i in range(len(mask_labels)):
                            mask_key = f'mask_{i}'
                            # Check if mask still has content after transformation
                            if mask_key in transformed and torch.any(transformed[mask_key] > 0):
                                valid_masks.append(transformed[mask_key])
                                valid_labels.append(mask_labels[i])
                    
                    # If we have valid masks or we're on the final attempt, use this transform
                    if len(valid_masks) > 0 or attempt == max_attempts - 1:
                        image = transformed['image']
                        if len(valid_masks) > 0:
                            masks = torch.stack(valid_masks)
                            mask_labels = torch.tensor(valid_labels, dtype=torch.long)
                        else:
                            # No valid masks remain
                            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.float)
                            mask_labels = torch.tensor([], dtype=torch.long)
                        break
                    
                    # If we got here, the transform eliminated all masks, so we'll try again
            else:
                # If no transform, convert numpy arrays to tensors
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                if len(masks) > 0:
                    masks = torch.from_numpy(masks.astype(np.float32))
                    mask_labels = torch.tensor(mask_labels, dtype=torch.long)
                else:
                    masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.float)
                    mask_labels = torch.tensor([], dtype=torch.long)

            return {
                'image': image,
                'masks': masks,
                'labels': mask_labels,
                'image_id': sample_path.name,
                'orig_height': orig_height,
                'orig_width': orig_width
            }
        
        else:
            # For test data, load only the image
            image_path = str(sample_path)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            filename = sample_path.name
            
            # Get original dimensions from metadata
            orig_height = None
            orig_width = None
            image_id = None
            
            if filename in self.image_metadata:
                metadata = self.image_metadata[filename]
                orig_height = metadata['height']
                orig_width = metadata['width']
                image_id = metadata['id']
            else:
                # If not in metadata, use the dimensions from the image itself
                orig_height, orig_width = image.shape[:2]
                image_id = filename
            
            # Store original dimensions and image before transform
            orig_image = image.copy()

            # Apply transforms if any
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return {
                'image': image,
                'image_id': image_id,
                'filename': filename,
                'orig_height': orig_height,
                'orig_width': orig_width,
                'orig_image': orig_image  # Store original image for reference
            }


def apply_transforms(transform, image, masks):
    """
    Apply albumentations transforms to image and multiple masks.
    
    Args:
        transform: Albumentations transform.
        image: Input image.
        masks: Array of masks [N, H, W].
        
    Returns:
        Dict with transformed image and masks.
    """
    # Prepare additional_targets for multiple masks
    additional_targets = {}
    for i in range(masks.shape[0]):
        additional_targets[f'mask_{i}'] = 'mask'
    
    # Create new transform with additional targets
    aug = A.Compose([t for t in transform], additional_targets=additional_targets)
    
    # Prepare inputs
    inputs = {'image': image}
    for i in range(masks.shape[0]):
        inputs[f'mask_{i}'] = masks[i]
    
    # Apply transforms
    return aug(**inputs)


def get_train_transforms(height=1024, width=1024, disable_spatial=False):
    """
    Get training transformations with Albumentations.
    
    Args:
        height (int): Target height.
        width (int): Target width.
        disable_spatial (bool): If True, disables spatial transformations that might affect masks.
        
    Returns:
        List of transformations.
    """

    return get_test_transforms(height, width)

    transforms_list = []
    
    if disable_spatial:
        # Only resize without additional spatial transformations
        transforms_list.append(A.Resize(height=height, width=width))
    else:
        # Regular spatial transforms
        transforms_list.extend([
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ])
    
    # Add non-spatial transforms that won't affect mask boundaries
    transforms_list.extend([
        # Color transforms that only modify the image
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
            A.GaussNoise(var_limit=(10.0, 50.0)),
        ], p=0.5),
        
        # Blur and sharpening
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
        ], p=0.3),
        
        # Medical image-specific augmentations (more subtle)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        
        # Normalize and convert to tensor (must be last)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(height=1024, width=1024):
    """
    Get validation transformations with Albumentations.
    Similar to test transforms but can be customized if needed.
    
    Args:
        height (int): Target height.
        width (int): Target width.
        
    Returns:
        List of transformations.
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def get_test_transforms(height=1024, width=1024):
    """
    Get test transformations with Albumentations.
    
    Args:
        height (int): Target height.
        width (int): Target width.
        
    Returns:
        List of transformations.
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


class MaskRCNNCollator:
    """
    Collator for Mask R-CNN model
    """
    def __call__(self, batch):
        """
        Collate function to prepare data for Mask R-CNN.
        
        Args:
            batch: List of samples from the dataset.
            
        Returns:
            A batch of samples with properly formatted targets.
        """
        images = []
        targets = []
        
        for sample in batch:
            images.append(sample['image'])
            
            target = {}
            
            # Include original dimensions in all targets
            if 'orig_height' in sample:
                target['orig_height'] = sample['orig_height']
            if 'orig_width' in sample:
                target['orig_width'] = sample['orig_width']
            
            # Include image ID in all targets
            target['image_id'] = sample['image_id']
            
            if 'masks' in sample:
                # Format targets for Mask R-CNN
                target.update({
                    'boxes': self._get_boxes(sample['masks']),
                    'labels': sample['labels'],
                    'masks': sample['masks']
                })
            else:
                # For test data, include all relevant info
                target.update({
                    'filename': sample['filename'],
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros(0, dtype=torch.long),
                    'masks': torch.zeros((0, sample['image'].shape[1], sample['image'].shape[2]), dtype=torch.float)
                })
                
                # Add original image for test data if available
                if 'orig_image' in sample:
                    target['orig_image'] = sample['orig_image']
            
            targets.append(target)
        
        return images, targets
    
    def _get_boxes(self, masks):
        """
        Convert masks to bounding boxes with validation to ensure proper dimensions.
        
        Args:
            masks (torch.Tensor): Binary masks of shape [N, H, W].
            
        Returns:
            torch.Tensor: Bounding boxes in format [N, 4] (x1, y1, x2, y2).
        """
        if masks.shape[0] == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        
        boxes = []
        for mask in masks:
            pos = torch.where(mask > 0)
            if len(pos[0]) == 0:
                # Empty mask
                boxes.append(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
                continue
                
            xmin = torch.min(pos[1])
            ymin = torch.min(pos[0])
            xmax = torch.max(pos[1])
            ymax = torch.max(pos[0])
            
            # Ensure the box has valid dimensions (at least 1 pixel width and height)
            if xmax == xmin:
                if xmax < masks.shape[1] - 1:
                    xmax = xmax + 1
                else:
                    xmin = xmin - 1
            
            if ymax == ymin:
                if ymax < masks.shape[0] - 1:
                    ymax = ymax + 1
                else:
                    ymin = ymin - 1
            
            boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
        
        return torch.stack(boxes)


def get_dataloader(root_dir, batch_size=2, split='train', num_workers=4, transform=None, 
                  height=1024, width=1024, disable_spatial_transforms=False, 
                  validation_split=0.2, seed=42):
    """
    Create a DataLoader for the medical cell dataset.
    
    Args:
        root_dir (str): Root directory for the dataset.
        batch_size (int): Batch size.
        split (str): 'train' or 'test' split.
        num_workers (int): Number of workers for data loading.
        transform (callable, optional): Transform to apply to the data.
        height (int): Target height for resizing.
        width (int): Target width for resizing.
        disable_spatial_transforms (bool): If True, disables spatial transformations in training.
        validation_split (float): Fraction of the dataset to use for validation (only used when split='train').
        seed (int): Random seed for reproducibility.
        
    Returns:
        If split is 'train', returns (train_loader, val_loader)
        If split is 'test', returns test_loader
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create custom collator for Mask R-CNN
    collator = MaskRCNNCollator()
    
    if split == 'train':
        # Set default train transforms if not provided
        train_transform = transform
        if train_transform is None:
            train_transform = get_train_transforms(height, width, disable_spatial=disable_spatial_transforms)
        
        # Create full training dataset
        full_dataset = MedicalCellDataset(root_dir=root_dir, split='train', transform=train_transform)
        
        # Calculate the size of each split
        dataset_size = len(full_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        print(f"Total dataset size: {dataset_size}")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        # Create training and validation sets
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # For validation, create a dataset with validation transforms
        # We need to create a custom wrapper to apply validation transforms
        val_transform = get_val_transforms(height, width)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )

        if validation_split == 0:
            # If no validation split, return only the training loader
            return train_loader, None
        
        # Create a validation dataset with val transforms
        # We need to wrap the val_dataset in a new dataset that applies the val_transform
        class ValDatasetWrapper(Dataset):
            def __init__(self, dataset, transform):
                self.dataset = dataset
                self.transform = transform
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                # Get the original sample
                sample = self.dataset[idx]
                
                # For validation, we need to apply validation transforms to the image
                # Extract the image as numpy array (convert from tensor if needed)
                if isinstance(sample['image'], torch.Tensor):
                    # Convert tensor back to numpy for new transform
                    image = sample['image'].permute(1, 2, 0).numpy()
                    # Denormalize if needed
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image = image * std + mean
                    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                else:
                    image = sample['image']  # Already numpy
                
                # Apply validation transform
                transformed = self.transform(image=image)
                sample['image'] = transformed['image']
                
                return sample
        
        # Only wrap with val transform if the dataset was created with train transforms
        if transform is None:
            val_dataset_wrapped = ValDatasetWrapper(val_dataset, val_transform)
        else:
            # If custom transform was provided, we'll use it for validation too
            val_dataset_wrapped = val_dataset
        
        val_loader = DataLoader(
            val_dataset_wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    else:  # Test split
        # Set default test transforms if not provided
        test_transform = transform
        if test_transform is None:
            test_transform = get_test_transforms(height, width)
        
        # Create test dataset
        test_dataset = MedicalCellDataset(root_dir=root_dir, split='test', transform=test_transform)
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
        
        return test_loader
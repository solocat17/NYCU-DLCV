import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

NUM_CLASSES = 100

class CutMixCollator:
    """
    Collator that applies CutMix augmentation to batches of images.
    
    CutMix randomly cuts a rectangular region from one image and pastes it onto another image,
    blending the labels proportionally to the area of the cut region.
    """

    def __init__(self, alpha=1.0, prob=0.5):
        """
        Args:
            alpha (float): Parameter for beta distribution to sample mixing ratio.
            prob (float): Probability of applying CutMix to a batch.
        """

        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        # Apply CutMix with probability self.prob
        if random.random() < self.prob:
            batch_size = len(images)
            
            # Sample mixing ratio from beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Randomly select an image to mix with
            rand_index = torch.randperm(batch_size)
            
            # Get target labels for the mixed pairs
            target_a = labels
            target_b = labels[rand_index]
            
            # Get image dimensions
            _, h, w = images[0].shape
            
            # Sample bounding box for the cutout area
            cut_ratio = np.sqrt(1. - lam)
            cut_w = int(w * cut_ratio)
            cut_h = int(h * cut_ratio)
            
            # Randomly select center point of cutout
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            # Calculate box boundaries
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            
            # Create CutMixed images
            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual cutout size
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            
            # Return mixed images and the pair of labels with their respective weights
            return mixed_images, target_a, target_b, lam
        
        # If not applying CutMix, return images and labels normally
        return images, labels, None, None


class ImageClassificationDataset(Dataset):
    """
    Dataset for image classification tasks with support for train, validation, and test sets.
    """

    def __init__(self, root_dir, transform=None, is_test=False):
        """
        Args:
            root_dir (str): Directory containing the dataset.
            transform (callable, optional): Optional transform to be applied to images.
            is_test (bool): Whether this is a test dataset without labels.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.image_paths = []
        self.labels = []
        
        if not is_test:
            # For training and validation sets
            for class_id in range(NUM_CLASSES):  # 0-99 classes
                class_dir = os.path.join(root_dir, str(class_id))
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.endswith('.jpg'):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(class_id)
        else:
            # For test set
            for img_name in os.listdir(root_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(root_dir, img_name))
                    # For test set, we don't have labels, use dummy labels
                    self.labels.append(0)  # Placeholder
    
    def __len__(self):
        """Returns the number of samples in the dataset."""

        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to return.
            
        Returns:
            tuple: (image, label) or (image, path) for test sets
        """

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            # For test set, return the image and its path for identification
            return image, img_path
        else:
            # For train/val, return image and class label
            return image, self.labels[idx]


def get_transforms(is_train=True):
    """
    Define data transformation pipelines for training and evaluation.
    
    Args:
        is_train (bool): Whether to return transformations for training or evaluation.
        
    Returns:
        transforms.Compose: Composition of image transformations.
    """
    
    # ImageNet mean and std for normalization when using pretrained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        # More aggressive augmentation for training
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Simple resize and normalization for validation/test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def get_data_loaders(train_dir, val_dir, test_dir=None, batch_size=32, num_workers=4, use_cutmix=True):
    """
    Create and return DataLoaders for training, validation, and optionally testing.
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        test_dir (str, optional): Directory containing test data
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of workers for DataLoaders
        use_cutmix (bool): Whether to use CutMix augmentation for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) - DataLoaders for each split
    """

    # Create datasets
    train_dataset = ImageClassificationDataset(
        root_dir=train_dir,
        transform=get_transforms(is_train=True),
        is_test=False
    )

    val_dataset = ImageClassificationDataset(
        root_dir=val_dir,
        transform=get_transforms(is_train=False),
        is_test=False
    )
    
    # CutMix collator for training
    cutmix_collator = CutMixCollator(alpha=1.0, prob=1.0) if use_cutmix else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cutmix_collator if use_cutmix else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Optionally create test loader
    test_loader = None
    if test_dir:
        test_dataset = ImageClassificationDataset(
            root_dir=test_dir,
            transform=get_transforms(is_train=False),
            is_test=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loading and visualization
    import matplotlib.pyplot as plt
    
    # Create datasets and loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        test_dir='data/test',
        batch_size=4,
        use_cutmix=True
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Function to denormalize images for visualization
    def denormalize(tensor):
        """Convert normalized image tensor to displayable image."""

        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def save_image(img, title, label):
        """Save a single image with title and label."""

        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        figtitle = f"{title}_{label}"
        plt.title(figtitle)
        plt.savefig(f'output/{figtitle}.png')
        plt.close()

    def show_images(loader, title, max_images=3):
        """Display and save images from a data loader."""
        
        os.makedirs('output', exist_ok=True)
        
        # Get a batch
        data = next(iter(loader))
        
        # Check if CutMix was applied
        if len(data) == 4:
            images, labels_a, labels_b, lam = data
            print(f"CutMix batch with lambda: {lam}")
            labels = [f"{labels_a[i]}-{labels_b[i]}" for i in range(min(max_images, len(images)))]
        else:
            images, labels = data
            if isinstance(labels[0], str):  # Test set returns paths
                labels = [os.path.basename(path) for path in labels[:max_images]]
            else:
                labels = labels[:max_images].tolist()
        
        # Process at most max_images
        for i in range(min(max_images, len(images))):
            img = denormalize(images[i])
            img = img.permute(1, 2, 0).numpy()  # CxHxW to HxWxC
            img = np.clip(img, 0, 1)
            save_image(img, f'{title}_{i}', labels[i])
        
        print(f"{title} images saved successfully!")

    # Visualize samples from each dataset
    show_images(train_loader, 'train')
    show_images(val_loader, 'val')
    if test_loader:
        show_images(test_loader, 'test')
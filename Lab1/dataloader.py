import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

NUM_CLASSES = 100

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
                    # For test set, use dummy labels as placeholders
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            # For test set, return the image and its path as a string for identification
            return image, img_path  # Return the actual path string, not self.labels[idx]
        else:
            # For train/val, return image and class label
            return image, self.labels[idx]

# Define data augmentation and normalization
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


if __name__ == "__main__":
    # Create datasets
    train_dataset = ImageClassificationDataset(
        root_dir='data/train',
        transform=get_transforms(is_train=True),
        is_test=False
    )

    val_dataset = ImageClassificationDataset(
        root_dir='data/val',
        transform=get_transforms(is_train=False),
        is_test=False
    )

    test_dataset = ImageClassificationDataset(
        root_dir='data/test',
        transform=get_transforms(is_train=False),
        is_test=True
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # plot and save 3 images from the training set, 3 images from the validation set, and 3 images from the test set
    import matplotlib.pyplot as plt
    import numpy as np

    def imsave(img, title, label):
        """
        Save an image with a title and label.
        
        Args:
            img: Image to save
            title (str): Base title for the image
            label: Label to append to the title
        """
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        figtitle = f"{title}_{label}"
        plt.title(figtitle)
        plt.savefig(f'output/{figtitle}.png')
        plt.close()

    def show_images(loader, title):
        for i, (images, labels) in enumerate(loader):
            if i == 3:
                break
            images = images[0]
            images = images * torch.tensor(std).view(3, 1, 1)
            images = images + torch.tensor(mean).view(3, 1, 1)
            img = np.transpose(images.numpy(), (1, 2, 0))
            imsave(img, f'{title}_{i}', labels[0].item())

    show_images(train_loader, 'train')
    show_images(val_loader, 'val')
    show_images(test_loader, 'test')
    
    print("Images saved successfully!")


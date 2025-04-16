import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np

class DigitRecognitionDataset(Dataset):
    """
    Dataset for Digit Recognition with both classification and bounding box tasks
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            split (string): 'train', 'valid', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set paths
        self.image_dir = os.path.join(root_dir, 'nycu-hw2-data', split)
        
        # Load annotations if not test set
        self.annotations = []
        self.img_to_annotations = {}
        self.images_info = []
        
        if split != 'test':
            annotation_file = os.path.join(root_dir, 'nycu-hw2-data', f'{split}.json')
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Extract category mapping
            self.categories = {cat['id']: int(cat['name']) for cat in data['categories']}
            
            # Extract image information
            self.images_info = {img['id']: img for img in data['images']}
            
            # Group annotations by image_id
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_to_annotations:
                    self.img_to_annotations[img_id] = []
                self.img_to_annotations[img_id].append(ann)
            
            # Create a list of image ids
            self.image_ids = list(self.images_info.keys())
        else:
            # For test set, we just list all images in the directory
            self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')], 
                                         key=lambda x: int(x.split('.')[0]))
    
    def __len__(self):
        if self.split != 'test':
            return len(self.image_ids)
        else:
            return len(self.image_filenames)
    
    def __getitem__(self, idx):
        if self.split != 'test':
            img_id = self.image_ids[idx]
            img_info = self.images_info[img_id]
            img_path = os.path.join(self.image_dir, img_info['file_name'])
            
            # Get image dimensions
            img_height = img_info['height']
            img_width = img_info['width']
            
            # Load annotations for this image
            annotations = self.img_to_annotations.get(img_id, [])
            
            # Extract bounding boxes and categories
            boxes = []
            categories = []
            
            for ann in annotations:
                # COCO format bbox is [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format for training
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                
                # Get digit class (0-9)
                category_id = ann['category_id']
                categories.append(category_id)
            
            # If no annotations, create empty tensors
            if not boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                categories = torch.zeros(0, dtype=torch.int64)
                digit_count = 0
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                categories = torch.tensor(categories, dtype=torch.int64)
                
                # Get the digit count (task 2) - extracting the number represented by the digits
                # Sort boxes from left to right
                if len(boxes) > 0:
                    sorted_indices = torch.argsort(boxes[:, 0])
                    categories = categories[sorted_indices]
                    boxes = boxes[sorted_indices]
                
                # Construct the number from the digits
                digits = [str(cat.item()) for cat in categories]
                digit_count = int(''.join(digits)) if digits else 0
            
        else:
            # For test set, we only load the image
            img_path = os.path.join(self.image_dir, self.image_filenames[idx])
            boxes = torch.zeros((0, 4), dtype=torch.float32)  # Placeholder
            categories = torch.zeros(0, dtype=torch.int64)    # Placeholder
            digit_count = 0  # Placeholder
        
        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.split == 'test':
            img_width, img_height = image.size

        # if self.split != 'test':
        new_width, new_height = 400, 400
        scale_width = new_width / image.width
        scale_height = new_height / image.height
        image = F.resize(image, [new_height, new_width])

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_width  # Scale x coordinates
            boxes[:, [1, 3]] *= scale_height  # Scale y coordinates
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Create a sample dictionary
        sample = {
            'image': image,
            'original_size': (img_width, img_height),
            'boxes': boxes,
            'categories': categories,
            'digit_count': digit_count,
            'image_id': img_id if self.split != 'test' else int(self.image_filenames[idx].split('.')[0])
        }
        
        return sample

def get_transform(train=True):
    """
    Get the transform for the dataset
    Args:
        train (bool): Whether to use train transforms or validation transforms
    """
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])

def collate_fn(batch):
    """
    Custom collate function for batches with variable number of bounding boxes
    """
    images = [item['image'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    categories = [item['categories'] for item in batch]
    digit_counts = [item['digit_count'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    # Stack images into a batch
    images = torch.stack(images, 0)
    
    # Return as a dictionary
    return {
        'images': images,
        'original_sizes': original_sizes,
        'boxes': boxes,
        'categories': categories,
        'digit_counts': torch.tensor(digit_counts),
        'image_ids': torch.tensor(image_ids)
    }

def get_dataloaders(root_dir, batch_size=16):
    """
    Get train, validation, and test dataloaders
    Args:
        root_dir (string): Root directory of the dataset
        batch_size (int): Batch size
    """
    # Create datasets
    train_dataset = DigitRecognitionDataset(
        root_dir=root_dir,
        split='train',
        transform=get_transform(train=True)
    )
    
    valid_dataset = DigitRecognitionDataset(
        root_dir=root_dir,
        split='valid',
        transform=get_transform(train=False)
    )
    
    test_dataset = DigitRecognitionDataset(
        root_dir=root_dir,
        split='test',
        transform=get_transform(train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


import matplotlib.pyplot as plt
import matplotlib.patches as patches
def plot_img_bbox(img, output, save_path=None):
    """Plot image with bounding boxes and save to file"""
    if torch.is_tensor(img):
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        x, y, x2, y2 = box
        width = x2 - x
        height = y2 - y

        # Create rectangle patch
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)

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

# Example usage:
if __name__ == "__main__":
    # Assuming the root directory is 'Lab2'
    root_dir = '.'
    
    # Get dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(root_dir)
    
    # Example of iterating through the dataloader
    for batch_idx, batch in enumerate(train_loader):
        # images is list, convert to tensor
        images = batch['images']
        boxes = batch['boxes']
        categories = batch['categories']
        digit_counts = batch['digit_counts']
        
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Number of boxes in first image: {len(boxes[0])}")
        print(f"  Categories in first image: {categories[0]}")
        print(f"  Digit count in first image: {digit_counts[0]}")

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        # Draw bounding boxes on images
        bounding_box_color = (255, 255, 255)
        for i, (image, box) in enumerate(zip(images, boxes)):
            image = image.permute(1, 2, 0).cpu().numpy()
            box = box.cpu().numpy()
            plot_img_bbox(image, {
                'boxes': box,
                'labels': categories[i],
                'scores': torch.ones(len(box))  # Dummy scores
            }, save_path=os.path.join(output_dir, f'image_{batch_idx}_{i}.png'))
        print(f"  Saved images to {output_dir}")

        break

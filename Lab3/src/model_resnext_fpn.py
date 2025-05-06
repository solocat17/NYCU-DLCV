import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import OrderedDict

class CustomBackboneWithFPN(nn.Module):
    """
    Custom ResNeXt101-32x8d with FPN backbone for Mask R-CNN.
    """
    def __init__(self, pretrained=True, out_channels=256):
        super(CustomBackboneWithFPN, self).__init__()
        
        # Load pretrained ResNeXt101
        resnext = torchvision.models.resnext101_32x8d(weights='DEFAULT' if pretrained else None)
        
        # Extract layers from ResNeXt101
        self.layer0 = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool
        )
        self.layer1 = resnext.layer1  # 256 channels
        self.layer2 = resnext.layer2  # 512 channels
        self.layer3 = resnext.layer3  # 1024 channels
        self.layer4 = resnext.layer4  # 2048 channels
        
        # FPN inner layers - reduce channel dimensions
        self.inner_layer1 = nn.Conv2d(256, out_channels, 1)
        self.inner_layer2 = nn.Conv2d(512, out_channels, 1)
        self.inner_layer3 = nn.Conv2d(1024, out_channels, 1)
        self.inner_layer4 = nn.Conv2d(2048, out_channels, 1)
        
        # FPN smoothing layers
        self.smooth_layer1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_layer2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_layer3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_layer4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Extra layer for P6 (used for RPN)
        self.extra_layer = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        # Output channels for compatibility with Mask R-CNN
        self.out_channels = out_channels
        
        # Initialize new layers
        for m in [self.inner_layer1, self.inner_layer2, self.inner_layer3, self.inner_layer4,
                 self.smooth_layer1, self.smooth_layer2, self.smooth_layer3, self.smooth_layer4,
                 self.extra_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Bottom-up pathway (ResNeXt101 backbone)
        c1 = self.layer0(x)       # 1/4 resolution
        c2 = self.layer1(c1)      # 1/4 resolution, 256 channels
        c3 = self.layer2(c2)      # 1/8 resolution, 512 channels
        c4 = self.layer3(c3)      # 1/16 resolution, 1024 channels
        c5 = self.layer4(c4)      # 1/32 resolution, 2048 channels
        
        # Top-down pathway and lateral connections (FPN)
        p5 = self.inner_layer4(c5)
        p4 = self.inner_layer3(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.inner_layer2(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.inner_layer1(c2) + nn.functional.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        
        # Apply smoothing
        p2 = self.smooth_layer1(p2)
        p3 = self.smooth_layer2(p3)
        p4 = self.smooth_layer3(p4)
        p5 = self.smooth_layer4(p5)
        
        # Extra feature map for larger objects
        p6 = self.extra_layer(p5)
        
        # Return feature pyramid as OrderedDict
        return OrderedDict([
            ('0', p2),  # 1/4 resolution
            ('1', p3),  # 1/8 resolution
            ('2', p4),  # 1/16 resolution
            ('3', p5),  # 1/32 resolution
            ('4', p6),  # 1/64 resolution (extra level)
        ])


class AttentionBlock(nn.Module):
    """
    Attention block for enhancing features at different scales.
    """
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compute attention weights
        attention_weights = self.attention(x)
        # Apply attention
        return x * attention_weights


class ResNeXt101EnhancedFPN(nn.Module):
    """
    Enhanced FPN with ResNeXt101 backbone and attention modules
    """
    def __init__(self, pretrained=True, out_channels=256):
        super(ResNeXt101EnhancedFPN, self).__init__()
        
        # Base backbone with FPN
        self.backbone_fpn = CustomBackboneWithFPN(pretrained=pretrained, out_channels=out_channels)
        
        # Attention modules for each FPN level
        self.attention_p2 = AttentionBlock(out_channels)
        self.attention_p3 = AttentionBlock(out_channels)
        self.attention_p4 = AttentionBlock(out_channels)
        self.attention_p5 = AttentionBlock(out_channels)
        
        # Output channels for compatibility with Mask R-CNN
        self.out_channels = out_channels
    
    def forward(self, x):
        # Get FPN features
        fpn_features = self.backbone_fpn(x)
        
        # Apply attention to each level
        enhanced_features = OrderedDict([
            ('0', self.attention_p2(fpn_features['0'])),
            ('1', self.attention_p3(fpn_features['1'])),
            ('2', self.attention_p4(fpn_features['2'])),
            ('3', self.attention_p5(fpn_features['3'])),
            ('4', fpn_features['4']),  # Keep P6 as is for RPN
        ])
        
        return enhanced_features


def get_maskrcnn_resnext101_fpn(num_classes=4, pretrained=True):
    """
    Create a Mask R-CNN model with a ResNeXt101-32x8d backbone and enhanced FPN.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights for the backbone
        
    Returns:
        model: Mask R-CNN model with ResNeXt101 backbone
    """
    # Create the custom backbone with FPN
    backbone = ResNeXt101EnhancedFPN(pretrained=pretrained)
    
    # Create Mask R-CNN with the custom backbone
    model = torchvision.models.detection.mask_rcnn.MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=512,
        max_size=1024
    )
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one (optional: can customize further)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


# Wrapper class for consistent API across model variants
class MaskRCNNResNeXt101Wrapper:
    """
    Wrapper class for Mask R-CNN with ResNeXt101 backbone to make it compatible with the inference pipeline.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def __call__(self, images):
        """
        Perform inference on a batch of images.
        
        Args:
            images (list): List of torch.Tensor images
            
        Returns:
            list: List of dictionaries containing predictions for each image
        """
        if not isinstance(images, list):
            images = [images]
            
        with torch.no_grad():
            # Move images to device
            images = [img.to(self.device) for img in images]
            
            # Get predictions
            predictions = self.model(images)
            
        return predictions
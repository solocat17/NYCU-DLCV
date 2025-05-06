import torch
import torch.nn as nn
import torchvision
import torchvision.models.detection.mask_rcnn as mask_rcnn
from collections import OrderedDict

class ResNeXtUNetBackbone(nn.Module):
    """
    Custom ResNeXt101-UNet hybrid backbone for Mask R-CNN.
    Combines the feature extraction power of ResNeXt with UNet-style skip connections.
    """
    def __init__(self, pretrained=True):
        super(ResNeXtUNetBackbone, self).__init__()
        
        # Load pretrained ResNeXt101
        self.resnext = torchvision.models.resnext101_32x8d(weights='DEFAULT' if pretrained else None)
        
        # Extract the layers we need from ResNeXt
        self.layer0 = nn.Sequential(
            self.resnext.conv1,
            self.resnext.bn1,
            self.resnext.relu,
            self.resnext.maxpool
        )
        self.layer1 = self.resnext.layer1  # 256 channels, /4 resolution
        self.layer2 = self.resnext.layer2  # 512 channels, /8 resolution
        self.layer3 = self.resnext.layer3  # 1024 channels, /16 resolution
        self.layer4 = self.resnext.layer4  # 2048 channels, /32 resolution
        
        # UNet-style decoder blocks (up-sampling path)
        self.up1 = UNetUpBlock(2048, 1024, 512)  # from layer4 to layer3
        self.up2 = UNetUpBlock(512, 512, 256)    # from up1 to layer2
        self.up3 = UNetUpBlock(256, 256, 128)    # from up2 to layer1
        self.up4 = UNetUpBlock(128, 64, 64)      # from up3 to layer0
        
        # Output feature maps to pass to FPN
        self.out_channels = 256
        
        # Feature projection layers to standardize channel dimensions
        self.c2_projection = nn.Conv2d(256, self.out_channels, kernel_size=1)
        self.c3_projection = nn.Conv2d(512, self.out_channels, kernel_size=1)
        self.c4_projection = nn.Conv2d(1024, self.out_channels, kernel_size=1)
        self.c5_projection = nn.Conv2d(2048, self.out_channels, kernel_size=1)
        self.p6_projection = nn.Conv2d(2048, self.out_channels, kernel_size=3, stride=2, padding=1)
        
        # UNet feature projections
        self.up1_projection = nn.Conv2d(512, self.out_channels, kernel_size=1)
        self.up2_projection = nn.Conv2d(256, self.out_channels, kernel_size=1)
        self.up3_projection = nn.Conv2d(128, self.out_channels, kernel_size=1)
        self.up4_projection = nn.Conv2d(64, self.out_channels, kernel_size=1)
        
        # Initialize the weights for the new layers
        for m in [self.up1, self.up2, self.up3, self.up4, 
                 self.c2_projection, self.c3_projection, self.c4_projection, 
                 self.c5_projection, self.p6_projection, 
                 self.up1_projection, self.up2_projection, 
                 self.up3_projection, self.up4_projection]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ResNeXt encoder forward pass
        c1 = self.layer0(x)     # 64 channels, /4 resolution
        c2 = self.layer1(c1)    # 256 channels, /4 resolution
        c3 = self.layer2(c2)    # 512 channels, /8 resolution
        c4 = self.layer3(c3)    # 1024 channels, /16 resolution
        c5 = self.layer4(c4)    # 2048 channels, /32 resolution
        
        # UNet decoder forward pass with skip connections
        u1 = self.up1(c5, c4)   # Using features from c5 and c4
        u2 = self.up2(u1, c3)   # Using features from u1 and c3
        u3 = self.up3(u2, c2)   # Using features from u2 and c2
        u4 = self.up4(u3, c1)   # Using features from u3 and c1
        
        # Project feature maps to consistent channel dimension for FPN
        p2 = self.c2_projection(c2)
        p3 = self.c3_projection(c3)
        p4 = self.c4_projection(c4)
        p5 = self.c5_projection(c5)
        p6 = self.p6_projection(c5)  # Extra feature level for larger objects
        
        # Project UNet features
        up1_feat = self.up1_projection(u1)
        up2_feat = self.up2_projection(u2)
        up3_feat = self.up3_projection(u3)
        up4_feat = self.up4_projection(u4)
        
        # Create feature map dictionary for FPN
        # Combine ResNeXt hierarchical features with UNet decoder features
        features = OrderedDict([
            ('0', p2),      # /4 resolution from ResNeXt
            ('1', p3),      # /8 resolution from ResNeXt
            ('2', p4),      # /16 resolution from ResNeXt
            ('3', p5),      # /32 resolution from ResNeXt
            ('4', p6),      # /64 resolution extra level
            ('unet1', up1_feat),  # UNet features at /16 resolution
            ('unet2', up2_feat),  # UNet features at /8 resolution
            ('unet3', up3_feat),  # UNet features at /4 resolution
            ('unet4', up4_feat),  # UNet features at /4 resolution
        ])
        
        return features


class UNetUpBlock(nn.Module):
    """
    Up-sampling block for the UNet-style decoder.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle cases where skip dimensions don't match (adjust if needed)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNeXtUNetWithFPN(nn.Module):
    """
    Full backbone with FPN integration to be used in Mask R-CNN.
    """
    def __init__(self, pretrained=True, out_channels=256):
        super(ResNeXtUNetWithFPN, self).__init__()
        
        # Initialize the base ResNeXt-UNet backbone
        self.resnext_unet = ResNeXtUNetBackbone(pretrained=pretrained)
        
        # Create extra FPN layers for improved feature fusion
        # Standard FPN layers
        self.fpn_inner_layer4 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        self.fpn_inner_layer3 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        self.fpn_inner_layer2 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        self.fpn_inner_layer1 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        
        # UNet enhanced FPN layers
        self.fpn_unet_layer3 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        self.fpn_unet_layer2 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        self.fpn_unet_layer1 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        self.fpn_unet_layer0 = nn.Conv2d(self.resnext_unet.out_channels, out_channels, 1)
        
        # Smoothing layers
        self.fpn_layer4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_layer3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_layer2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_layer1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_layer0 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Layer for p6 (additional level for larger objects)
        self.fpn_p6 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        # Set up output channels for compatibility with Mask R-CNN
        self.out_channels = out_channels
    
    def forward(self, x):
        # Get multi-scale features from ResNeXt-UNet
        features = self.resnext_unet(x)
        
        # Standard FPN top-down pathway
        # Start with the deepest layer
        p5 = self.fpn_inner_layer4(features['3'])
        p4 = self.fpn_inner_layer3(features['2']) + nn.functional.interpolate(p5, size=features['2'].shape[-2:], mode='nearest')
        p3 = self.fpn_inner_layer2(features['1']) + nn.functional.interpolate(p4, size=features['1'].shape[-2:], mode='nearest')
        p2 = self.fpn_inner_layer1(features['0']) + nn.functional.interpolate(p3, size=features['0'].shape[-2:], mode='nearest')
        
        # UNet-enhanced pathway - incorporate UNet decoded features
        p4 = p4 + self.fpn_unet_layer3(features['unet1'])
        p3 = p3 + self.fpn_unet_layer2(features['unet2'])
        p2 = p2 + self.fpn_unet_layer1(features['unet3'])
        
        # Additional smoothing
        p5 = self.fpn_layer4(p5)
        p4 = self.fpn_layer3(p4)
        p3 = self.fpn_layer2(p3)
        p2 = self.fpn_layer1(p2)
        
        # Generate p6 for detecting larger objects
        p6 = self.fpn_p6(p5)
        
        # Return feature pyramid
        return OrderedDict([
            ('0', p2),  # /4 resolution
            ('1', p3),  # /8 resolution
            ('2', p4),  # /16 resolution
            ('3', p5),  # /32 resolution
            ('4', p6),  # /64 resolution (extra level)
        ])


def get_maskrcnn_resnext101_unet(num_classes=4, pretrained=True):
    """
    Create a Mask R-CNN model with a custom ResNeXt101-UNet hybrid backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights for the backbone
        
    Returns:
        model: Mask R-CNN model with ResNeXt101-UNet backbone
    """
    # Create the custom backbone with FPN
    backbone = ResNeXtUNetWithFPN(pretrained=pretrained)
    
    # Create Mask R-CNN model with the custom backbone
    model = mask_rcnn.MaskRCNN(
        backbone,
        num_classes=num_classes,
        min_size=512,
        max_size=1024
    )
    
    return model
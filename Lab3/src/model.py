import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.model_resnext_fpn import get_maskrcnn_resnext101_fpn, MaskRCNNResNeXt101Wrapper
from src.model_resnext_unet import get_maskrcnn_resnext101_unet

def get_maskrcnn_model(num_classes=5, pretrained=True):
    """
    Create a Mask R-CNN model for instance segmentation.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: Mask R-CNN model
    """
    # Load pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights='DEFAULT' if pretrained else None,
        progress=True,
        min_size=512,
        max_size=1024
    )
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def get_maskrcnn_model_fpnv2(num_classes=5, pretrained=True):
    """
    Create a Mask R-CNN model with FPN v2 backbone for instance segmentation.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: Mask R-CNN model with FPN v2 backbone
    """
    # Load pre-trained model with FPN v2 backbone
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights='DEFAULT' if pretrained else None,
        progress=True,
        min_size=512,
        max_size=1024
    )
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


class MaskRCNNWrapper:
    """
    Wrapper class for Mask R-CNN model to make it compatible with the inference pipeline.
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
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn.functional as F

class DigitDetector(nn.Module):
    def __init__(self, num_classes=11):  # 10 digits (0-9) + background
        super(DigitDetector, self).__init__()
        
        # Set up the backbone with proper out_channels attribute
        backbone = resnet_fpn_backbone(
            backbone_name="resnet152",
            weights=torchvision.models.ResNet152_Weights.DEFAULT
        )
        
        # Define the anchor generator optimized for digit detection
        anchor_sizes = ((16,), (32,), (64,), (96,), (128,)) # Smaller anchors for digit detection
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) # Common aspect ratios for digits
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
                
        # Define the RoI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the Faster R-CNN model with ResNeXt50 backbone
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=400,
            max_size=400,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=1000,
            rpn_post_nms_top_n_test=500,
            rpn_nms_thresh=0.7,
            box_score_thresh=0.3,  # 0.5 Confidence threshold for detections
            box_nms_thresh=0.3    # 0.45 Non-max suppression threshold
        )
        
        # Replace the classifier with a new one for our digit classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    @staticmethod
    def convert_targets(boxes, labels, image_ids=None):
        """
        Convert the dataset format to Faster R-CNN expected format
        """
        targets = []
        for i, (box, label) in enumerate(zip(boxes, labels)):
            target = {}
            target["boxes"] = box
            target["labels"] = label
            if image_ids is not None:
                target["image_id"] = image_ids[i]
            targets.append(target)
        return targets

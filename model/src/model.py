import torch
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 11  # 10 classes + background


def get_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Returns Faster R-CNN model with ResNet50-FPN backbone.

    Args:
        num_classes (int): Number of detection classes including background.
        pretrained (bool): Whether to load COCO pretrained weights.

    Returns:
        torch.nn.Module: Faster R-CNN model
    """

    # model = fasterrcnn_resnet50_fpn(pretrained=pretrained)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Get input features of classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace head with new classifier
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

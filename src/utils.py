import logging
import numpy as np
import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights

logger = logging.getLogger('utils_loggers')


def create_model(output_channels: int = 1):
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progrss=True)
    model.classifier = DeepLabHead(2048, output_channels)
    logger.info('Instantiated model.')
    return model


def decode(logits: torch.Tensor, dim: int = 0) -> torch.Tensor:
    indeces = torch.argmax(logits, dim=dim, keepdim=True)
    result = torch.zeros_like(logits, dtype=torch.uint8)
    return result.scatter(dim=dim, index=indeces, value=1)
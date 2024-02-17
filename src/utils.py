import logging
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

logger = logging.getLogger('utils_loggers')


def create_model(output_channels: int = 1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progrss=True)
    model.classifier = DeepLabHead(2048, output_channels)
    logger.info('Instantiated model.')
    return model
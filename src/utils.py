import logging
import logging.config
import pathlib, gdown, zipfile, json, random, cv2
import torch
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights

logger = logging.getLogger('utils_loggers')


def create_model(output_channels: int = 1, weights: pathlib.Path | None = None):
    if not weights:
        model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1, progress=True, aux_loss=True)
        model.classifier = DeepLabHead(2048, output_channels)
    else:
        model = torch.load(weights, map_location=lambda loc, state: loc)
    model.eval()
    logger.info('Instantiated model.')
    return model


def decode(logits: torch.Tensor, dim: int = 0) -> torch.Tensor:
    indeces = torch.argmax(logits, dim=dim, keepdim=True)
    result = torch.zeros_like(logits, dtype=torch.uint8)
    return result.scatter(dim=dim, index=indeces, value=1)


def setup_data(url: str):
    logger.info('Downloading data...')
    datasets_root = pathlib.Path('datasets')
    dataset_folder = datasets_root.joinpath('CelebAMask-HQ')
    dataset_zipfile = datasets_root.joinpath('CelebAMask-HQ.zip')
    
    if not dataset_zipfile.exists():
        gdown.download(
            url=url, 
            output=str(dataset_zipfile),
            use_cookies=False,
            quiet=True
        )
    else:
        logger.info('Exisiting zipfile is used.')

    if not dataset_folder.exists():
        with zipfile.ZipFile(str(dataset_zipfile)) as zip:
            zip.extractall(path=dataset_folder)
    else:
        logger.info('Detected existing dataset. Loading it...')
    logger.info('Data loading ended.')


def setup_logging():
    logging_directory = pathlib.Path('logs', 'application')
    if not logging_directory.exists():
        logging_directory.mkdir(parents=True)
    config_path = pathlib.Path('src', 'logger', 'config.json')
    with config_path.open() as config_file:
        config = json.load(config_file)
    logging.config.dictConfig(config=config)


def seed_everything(seed: int = 0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    random.seed(seed)


def construct_image(output: torch.Tensor, mapping: dict, color_mapping: dict) -> np.ndarray:
    output_shape = (*output.shape[1:], 3)
    image = np.zeros(shape=output_shape)
    decoded_output = decode(logits=output)
    for layer_index in mapping.keys():
        index = int(layer_index)
        _raw_layer = np.stack([decoded_output[index], decoded_output[index], decoded_output[index]]).transpose(1, 2, 0)
        layer = _raw_layer * np.full(shape=output_shape, 
                                     fill_value=color_mapping[layer_index])
        image = image + layer
    return image

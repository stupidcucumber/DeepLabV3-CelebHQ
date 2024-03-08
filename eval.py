import argparse, pathlib, json, logging
import cv2
import torch
from torchvision import transforms
from src.utils import construct_image, create_model, setup_logging


logger = logging.getLogger('eval')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', type=pathlib.Path, required=True,
                        help='Path to the mapping')
    parser.add_argument('--model', type=pathlib.Path, default=None,
                        help='Path to the model weights.')
    parser.add_argument('-i', '--input', type=pathlib.Path, required=True,
                        help='Path to the input image.')
    parser.add_argument('-cmap', '--color-map', type=pathlib.Path, required=True,
                        help='Path to the color map.')
    parser.add_argument('-o', '--output', type=pathlib.Path, default=pathlib.Path('output.png'),
                        help='Path to the result file.')
    return parser.parse_args()


def load_input(image_path: pathlib.Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.Resize(size=(512, 512))
        ]
    )
    image = cv2.imread(str(image_path))
    shape = image.shape[:2][::-1]
    input = transform(image)
    return torch.unsqueeze(input, dim=0), shape


def load_mapping(mapping_path: pathlib.Path) -> dict:
    with mapping_path.open() as mapping_f:
        result = json.load(mapping_f)
    return result


if __name__ == '__main__':
    setup_logging()
    args = parse_arguments()
    
    input_image, init_shape = load_input(image_path=args.input)
    logger.info('Loaded input image.')
    mapping = load_mapping(mapping_path=args.mapping)
    color_map = load_mapping(mapping_path=args.color_map)
    logger.info('Loaded mapping')
    model = create_model(output_channels=len(mapping.keys()), weights=args.model)
    logger.info('Instantiated model.')

    logger.debug('Start inferencing.')
    model_output = model(input_image)
    segmentation = model_output['out']
    logger.debug('Ended inferencing.')
    output_image = construct_image(output=segmentation[0], mapping=mapping, color_mapping=color_map)
    output_image = cv2.resize(output_image, dsize=init_shape)
    logger.debug('Constructed image.')
    cv2.imwrite(str(args.output), output_image)
    logger.info('Saved image.')

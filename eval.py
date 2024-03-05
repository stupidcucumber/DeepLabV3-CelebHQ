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
    parser.add_argument('--model', type=pathlib.Path, required=True,
                        help='Path to the model weights.')
    parser.add_argument('-i', '--input', type=pathlib.Path, required=True,
                        help='Path to the input image.')
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
    input = transform(image)
    return input


def load_mapping(mapping_path: pathlib.Path) -> dict:
    with mapping_path.open() as mapping_f:
        result = json.load(mapping_f)
    return result


if __name__ == '__main__':
    setup_logging()
    args = parse_arguments()
    
    input_image = load_input(image_path=args.input)
    logger.info('Loaded input image.')
    mapping = load_mapping(mapping_path=args.mapping)
    logger.info('Loaded mapping')
    model = create_model(output_channels=len(mapping.keys()), weights=args.model)
    logger.info('Instantiated model.')

    logger.debug('Start inferencing.')
    output = model(torch.as_tensor([input_image]))
    logger.debug('Ended inferencing.')
    output_image = construct_image(output=output, mapping=mapping)
    logger.debug('Constructed image.')
    cv2.imwrite(str(args.output), output_image)
    logger.info('Saved image.')

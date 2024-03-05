import argparse
import pathlib
import cv2


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


if __name__ == '__main__':
    args = parse_arguments()
    

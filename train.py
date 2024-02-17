import logging, logging.config
import pathlib, json, argparse
from src.utils import create_model
from src import Trainer
import torch


logger = logging.getLogger('training_logger')


def setup_logging():
    logging_directory = pathlib.Path('logs')
    if not logging_directory.exists():
        logging_directory.mkdir()
    config_path = pathlib.Path('src/logger/config.json')
    with config_path.open() as config_file:
        config = json.load(config_file)
    logging.config.dictConfig(config=config)


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-channels', type=int, default=18)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    logger.info(
        'Parsed arguments.', 
        extra={key: value for key, value in args._get_kwargs()}
    )
    return args


if __name__ == '__main__':
    setup_logging()
    args = parse_configs()
    model = create_model(output_channels=args.output_channels)

    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model=model, 
                      loss_fn=loss_fn, 
                      optimizer=optimizer, 
                      logger=logger)

    trainer.fit(
        train_loader=None,
        val_loader=None,
        epochs=args.epochs
    )
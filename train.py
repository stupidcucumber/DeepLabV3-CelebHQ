import logging
import logging.config
import pathlib
import json

logger = logging.getLogger('training_logger')


def setup_logging():
    logging_directory = pathlib.Path('logs')
    if not logging_directory.exists():
        logging_directory.mkdir()
    config_path = pathlib.Path('src/logger/config.json')
    with config_path.open() as config_file:
        config = json.load(config_file)
    logging.config.dictConfig(config=config)


if __name__ == '__main__':
    setup_logging()
    
    logger.info('I want to break free!')
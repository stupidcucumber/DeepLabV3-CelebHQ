import logging, logging.config
import pathlib, json, argparse, gdown, zipfile, sys
from src.utils import create_model, setup_data, setup_logging
from src.data import SemanticDataset
from src.evaluators import AccuracyMeanEvaluator
from src.callback import BestWeightsCallback
from src import Trainer
import torch
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader


logger = logging.getLogger('train_script')

def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataframe that represents data. More in datasets/README.md')
    parser.add_argument('--output-channels', type=int, default=19)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tv-split', type=float, default=0.90,
                        help='Train-validation split, that defines the size of the train part.')
    parser.add_argument('--mapping', type=str, required=True,
                        help='Path to the mapping of the layer: index in json format')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='The device on which model will train.')
    args = parser.parse_args()
    if args.mapping:
        with open(args.mapping, 'r') as _mapping:
            args.mapping = {int(key): value for key, value in json.load(_mapping, parse_int=int).items()}
    logger.info(
        'Parsed arguments.', 
        extra={key: value for key, value in args._get_kwargs()}
    )
    return args


if __name__ == '__main__':
    setup_logging()
    setup_data(url='https://drive.google.com/uc?id=17e_IRjSuise59WUDHVrwZKES71KzJ9bU')
    args = parse_configs()
    model = create_model(output_channels=args.output_channels).to(args.device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
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
    evaluators = [
        AccuracyMeanEvaluator(name='accuracy',
                              mapping=args.mapping)
    ]
    callbacks = [
        BestWeightsCallback(
            output=pathlib.Path('runs')
        )
    ]

    trainer = Trainer(model=model, 
                      loss_fn=loss_fn, 
                      optimizer=optimizer,
                      evaluators=evaluators,
                      callbacks=callbacks,
                      device=args.device)

    data = pd.read_csv(args.data)
    split_index = int(len(data) * args.tv_split)
    train_data = data.iloc[:split_index].reset_index(drop=True)
    val_data = data.iloc[split_index:].reset_index(drop=True)
    train_dataset = SemanticDataset(
        df=train_data,
        seed=0,
        transforms=transform
    )
    val_dataset = SemanticDataset(
        df=val_data,
        seed=0,
        transforms=transform
    )
    logger.info('Loaded all datasets.')

    trainer.fit(
        train_loader=DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True),
        val_loader=DataLoader(dataset=val_dataset, batch_size=args.batch_size),
        epochs=args.epochs
    )
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from torchvision.transforms import ToTensor
import torch


class SemanticDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seed: int | None = None, transforms: None = None) -> None:
        super().__init__()
        self.seed = seed
        self.data = df
        self.transforms = transforms
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.data)
    
    def _construct_label(self, paths) -> list:
        result = []
        for path in paths:
            if pd.isna(path):
                dummy = np.full(shape=(512, 512), fill_value=0, dtype=np.int32)
                result.append(dummy)
                continue
            raw_image = cv2.imread(str(path))
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(raw_image, thresh=127, maxval=1, type=cv2.THRESH_BINARY)
            result.append(thresh)
        return result

    def __getitem__(self, index: int):
        entry = self.data.iloc[index]
        image = Image.open(entry['image_path'])
        print(entry['image_path'])
        input = self.transforms(image)
        mask_paths = entry.tolist()[1:]
        label = np.asarray([self.totensor(image) for image in self._construct_label(paths=mask_paths)])

        return input, torch.as_tensor(label).squeeze(dim=1)
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
    
    def _construct_label(self, paths) -> np.ndarray:
        '''
            This algorithm has strict order of semantic segmentation elements. For CelebAMask this order is:
        [neck -> cloth -> skin -> ears -> nose -> eyes -> brows -> mouth -> lips -> ears -> jewelery -> hair -> hat]
        '''
        result = np.full(shape=(512, 512), fill_value=0, dtype=np.int64)
        for index, path in enumerate(paths):
            if pd.isna(path):
                continue
            raw_image = cv2.imread(str(path))
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(raw_image, thresh=127, maxval=1, type=cv2.THRESH_BINARY)
            result[thresh > 0] = index
        return result

    def __getitem__(self, index: int):
        entry = self.data.loc[index]
        image = cv2.imread(entry['image_path'])
        input = self.transforms(image)
        mask_paths = entry.tolist()[1:]
        label = self._construct_label(paths=mask_paths)
        return input, torch.as_tensor(label, dtype=torch.long)
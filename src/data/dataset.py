from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class SemanticDataset(Dataset):
    def __init__(self, df_path: str, seed: int | None = None) -> None:
        super().__init__()
        self.seed = seed
        data = pd.read_csv(df_path)

        self.data = data.sample(frac=1).reset_index()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        entry = self.data.iloc[index]
        inputs = None
        labels = None

        return inputs, labels 
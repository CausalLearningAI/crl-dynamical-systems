from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .climate import SSTDataset
from .speedy_weather import SpeedyWeatherDataset

__all__ = [
    SSTDataset,
    SpeedyWeatherDataset,
]


# ---------------- pl DataModule ----------------
class LitDataModule(LightningDataModule):
    def __init__(self, train_set, val_set, test_set, batch_size: int = 1, num_workers=8, **kwargs):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        if hasattr(self.train_set, "collate_fn"):
            collate_fn = self.train_set.collate_fn
        else:
            collate_fn = None
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return test_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=self.val_set.predict_collate_fn,
        )
        return pred_loader

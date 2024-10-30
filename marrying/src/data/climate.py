import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------ real world sea surface temperature ------------------------
class SSTDataset(Dataset):
    """
    A PyTorch dataset class for handling SST (Sea Surface Temperature) data.

    Args:
        data_path (str): The path to the SST data.
        chunk_size (int): The size of each data chunk.
        mode (str): The mode of the dataset, either "train", "test" or "validation".

    Attributes:
        data (dict): A dictionary containing the SST data.
        time_steps (int): The number of time steps in the data.
        chunk_size (int): The size of each data chunk.
        lat_dim (int): The dimension of latitude in the data.
        lon_dim (int): The dimension of longitude in the data.
        mode (str): The mode of the dataset, either train", "test" or "validation".

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns a specific sample from the dataset.
        collate_fn(batch): Collates a batch of data samples.
        predict_collate_fn(batch): Collate function for forecasting on the last chunk of the data.
    """

    def __init__(
        self,
        data_path: str = "/nfs/scistore19/locatgrp/dyao/DATA/sst",
        chunk_size: int = 52 * 4,
        mode="train",
    ) -> None:
        super().__init__()
        from scipy.io import netcdf

        sst_netcdf = netcdf.NetCDFFile(os.path.join(data_path, "sst.wkmean.1990-present.nc"), "r")
        keys = list(sst_netcdf.variables.keys())
        self.data = {k: np.asarray(sst_netcdf.variables[k][:].byteswap().newbyteorder()) for k in keys}

        m, std = self.data["sst"].mean(), self.data["sst"].std()
        self.data["sst"] = (self.data["sst"] - m) / std  # [ts=1727, lat=180, lon=360]

        # time steps
        self.time_steps = self.data["sst"].shape[0]  # = 1727 #, too large
        self.chunk_size = chunk_size or self.time_steps
        self.lat_dim = self.data["lat"].shape[0]
        self.lon_dim = self.data["lon"].shape[0]

        # normalise time
        self.data["time"] = self.data["time"] - self.data["time"][0]
        self.data["sst"] = self.data["sst"].reshape(self.time_steps, -1).T
        self.mode = mode
        self.data["sst"] = self.data["sst"][..., None]

    def __len__(self):
        return self.data["sst"].shape[0]  # number of locations

    def __getitem__(self, index) -> Tuple:
        if self.mode != "test":
            if self.time_steps - 2 * self.chunk_size > 0:
                # randomly sample chunks from the training set. Last chunk kept for validation
                time_index = np.random.randint(self.time_steps - 2 * self.chunk_size)
            else:
                time_index = 0

            return {
                "index": index,
                "time_index": time_index,
                "states": self.data["sst"][index, time_index : time_index + self.chunk_size],
            }
        else:
            return {
                "index": index,
                "time_index": self.time_steps - 2 * self.chunk_size,
                "states": self.data["sst"][index, -2 * self.chunk_size :],
            }

    def collate_fn(self, batch: List[Dict]):
        """
        Collates a batch of data samples.

        Args:
            batch (List[Dict]): A list of dictionaries representing the data samples.
            Each dictionary should contain the following keys:
                - "index": The index of the sample.
                - "states": The states of the sample.
                - "time_index": The time index of the sample.

        Returns:
            dict: A dictionary containing the collated batch of data samples.
            The dictionary has the following keys:
                - "index": A tensor of shape (2, batch_size) representing the indices of the samples
                    and their augmented indices.
                - "states": A tensor of shape (2, batch_size, num_states) representing the states of the samples
                    and their augmented states.
        """
        indices = []
        states = []
        aug_indices = []
        aug_states = []
        time_indices = []
        for b in batch:
            indices += [b["index"]]
            states += [b["states"]]
            time_indices += [b["time_index"]]
            lat, lon = np.unravel_index(b["index"], (self.lat_dim, self.lon_dim))
            aug_lon = np.random.randint(max(0, lon - 5), min(self.lon_dim, lon + 5))  # index for sampled longitude
            aug_index = np.ravel_multi_index((lat, aug_lon), (self.lat_dim, self.lon_dim))
            aug_indices += [aug_index]
            aug_states += [self.data["sst"][aug_index, b["time_index"] : b["time_index"] + self.chunk_size]]

        batch_dict = {
            "index": torch.stack([torch.tensor(indices), torch.tensor(aug_indices)], dim=0),
            "states": torch.stack([torch.from_numpy(np.stack(states)), torch.from_numpy(np.stack(aug_states))], dim=0),
        }
        return batch_dict

    def predict_collate_fn(self, batch):
        """
        Collate function for forecasting on the last chrunk of the data.

        Args:
            batch (list): A list of dictionaries containing the batch data.

        Returns:
            dict: A dictionary containing the collated batch data with the following keys:
                - "index": A tensor containing the indices of the data samples in the batch.
                - "states": A tensor containing the stacked states for each data sample in the batch.
        """
        indices = []
        states = []
        time_indices_w_forecast = np.arange(self.time_steps - 2 * self.chunk_size, self.time_steps)
        for b in batch:
            indices += [b["index"]]
            states += [self.data["sst"][b["index"], time_indices_w_forecast]]

        batch_dict = {"index": torch.tensor(indices), "states": torch.from_numpy(np.stack(states))}
        return batch_dict


# class SSTDataset(Dataset):
#     def __init__(self, data_path: str, mode="train", start_year=1990) -> None:
#         super().__init__()
#         from scipy.io import netcdf

#         sst_netcdf = netcdf.NetCDFFile(data_path, "r")
#         keys = list(sst_netcdf.variables.keys())
#         self.data = {k: np.asarray(sst_netcdf.variables[k][:].byteswap().newbyteorder()) for k in keys}

#         m, std = self.data["sst"].mean(), self.data["sst"].std()
#         self.data["sst"] = (self.data["sst"] - m) / std  # [ts=1727, lat=180, lon=360]

#         # time steps
#         self.time_steps = 52 * 4  # self.data['sst'].shape[0] # = 1727 #, too large
#         self.lat_dim = self.data["lat"].shape[0]
#         self.lon_dim = self.data["lon"].shape[0]

#         # year_idx = start_year - 1990  # 1990 is the starting year of the data, maybe better to make it less
#         # self.time_indices = np.arange(year_idx * 52, year_idx * 52 + self.time_steps)
#         self.time_indices = np.arange(start_year, start_year + 52 * 4)
#         # normalise time
#         self.data["time"] = self.data["time"] - self.data["time"][0]

#         self.data["sst"] = self.data["sst"][self.time_indices].reshape(self.time_steps, -1).T
#         # self.data['sst'] = StandardScaler().fit_transform(self.data["sst"])

#         if mode in ["train"]:
#             self.data["sst"] = self.data["sst"][: int(0.9 * self.data["sst"].shape[0])]
#         elif mode in ["test"]:
#             self.data["sst"] = self.data["sst"][int(0.9 * self.data["sst"].shape[0]) :]
#         # for validation: to reproduce everything use the whole dataset

#         # what are the potential shared features here? across different regions?
#         # Can you split the trajectories into certain regions and require different part share different dimensions?

#     def __len__(self):
#         return self.data["sst"].shape[0]  # number of locations

#     def __getitem__(self, index) -> Tuple:
#         return {"index": index, "states": self.data["sst"][index][..., None]}
# # index of the location (i,j) and (i',j'

#     # TODO: define different collate function to specify pairs
#     def collate_fn(self, batch: List[Dict]):
#         indices = []
#         states = []
#         aug_indices = []
#         aug_states = []
#         for b in batch:
#             indices += [b["index"]]
#             states += [b["states"]]
#             lat, lon = np.unravel_index(b["index"], (self.lat_dim, self.lon_dim))
#             aug_lon = np.random.randint(max(0, lon - 5), min(self.lon_dim, lon + 5))  # index for sampled longitude
#             aug_index = np.ravel_multi_index((lat, aug_lon), (self.lat_dim, self.lon_dim))
#             aug_indices += [aug_index]
#             aug_states += [self.data["sst"][aug_index]]

#         batch_dict = {
#             "index": torch.stack([torch.tensor(indices), torch.tensor(aug_indices)], dim=0),
#             "states": torch.stack(
#                 [torch.from_numpy(np.stack(states)), torch.from_numpy(np.stack(aug_states)).unsqueeze(-1)], dim=0
#             ),
#         }
#         return batch_dict

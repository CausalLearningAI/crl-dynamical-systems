import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from data.generate_speedy import PARAM_SPACES
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm


############################## wind simulation for discretized factors ##############################
class SpeedyWeatherDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        model_name: str = "ShallowWaterModel",
        num_simulations: int = 100,
        mode: str = "train",
        n_views=5,
        include_keys=["u", "v", "vor"],
        shared_ids: List[int] = None,
        factor_sharing: Dict[Tuple, Tuple] = None,
        collate_style: str = "default",
        chunk_size=121,
        grid_size: List[int] = [2] * 4,
    ) -> None:
        """
        Initialize the SpeedyWeatherDataset object.

        Args:
            data_path (str): The path to the data directory.
            model_name (str, optional): The name of the model. Defaults to "ShallowWaterModel".
            num_simulations (int, optional): The number of simulations. Defaults to 100.
            mode (str, optional): The mode of the dataset (train, val, or test). Defaults to "train".
            n_views (int, optional): The number of views. Must be greater than 1. Defaults to 5.
            include_keys (List[str], optional): The keys to include in the dataset. Defaults to ["u", "v", "vor"].
            shared_ids (List[Union[int, List]], optional): The shared indices (the underlying causal graph).
                                                            Defaults to None.
            factor_sharing (Dict[int, List[int]], optional): The factor sharing configuration. Defaults to None.
            collate_style (str, optional): The collate style for the dataset. Defaults to "default".
            chunk_size (int, optional): The chunk size. Defaults to 121.
        """
        super().__init__()
        self.data_dir = os.path.join(data_path, model_name)
        assert os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."

        self.mode = mode
        assert mode in ["train", "val", "test"]

        self.n_views = n_views

        self.simulation_ids = range(num_simulations)
        self.include_keys = include_keys  # , "vor"]

        self.num_simulations = num_simulations
        self.initialized = False

        # define parameter space according to the simulated model
        if "ShallowWaterModel" in model_name:
            self.param_spaces = {
                kk: vv for k, v in PARAM_SPACES.items() for kk, vv in v.items() if kk == "layer_thickness"
            }
        elif "PrimitiveWetModel" in model_name:
            self.param_spaces = {
                kk: vv for k, v in PARAM_SPACES.items() for kk, vv in v.items() if kk != "layer_thickness"
            }
        else:
            raise ValueError(f"Model name {model_name} is not supported.")
        self.grid_size = grid_size  # defined while generating code
        self.__preprocess_data__()

        # split timestes into chrunks when trajectory is too long
        if chunk_size < self.time_steps // 2:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = self.time_steps

        # collate style for training
        assert collate_style in ["default", "random"], "collate_style must be either default or random"
        self.collate_stype = collate_style
        if collate_style == "default":
            self.shared_ids: List[int] = shared_ids
            self.factor_sharing: Dict[Tuple, Tuple] = factor_sharing
            assert shared_ids is not None, "shared_ids for the dataset must be provided"
            assert factor_sharing is not None, "factor_sharing for the dataset must be provided"
            self.collate_fn = self.default_collate_fn

        if collate_style == "random":
            assert self.n_views == 2, "random_collate_fn is only implemented for 2 views"
            self.collate_fn = self.random_collate_fn

    def __preprocess_data__(self):
        """
        Preprocess the data, save everything to self.data.
        """
        for i in tqdm(self.simulation_ids):
            file_path = os.path.join(self.data_dir, f"run_{i+1:04d}/output.nc")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist.")
            ds = h5py.File(file_path, "r")

            if not self.initialized:
                self.lat_dim = np.asarray(ds["lat"][:]).shape[0]
                self.lon_dim = np.asarray(ds["lon"][:]).shape[0]
                self.level_dim = np.asarray(ds["lev"][:]).shape[0]
                self.time_steps = np.asarray(ds["time"][:]).shape[0]
                self.data = {k: np.asarray(ds[k][:]) for k in ds.keys()}

                for k in self.include_keys:
                    self.data[k] = [self.data[k]]
                # voriticity: shape [time, level, lat, lon]
            else:
                for k in self.include_keys:
                    # stack the states
                    self.data[k] += [np.asarray(ds[k][:])]  # shape [time, level, lat, lon]
            self.initialized = True

        for k in self.include_keys:
            self.data[k] = np.stack(self.data[k], axis=0)
            self.data[k] = (
                StandardScaler()
                .fit_transform(self.data[k].reshape(-1, 1))
                .reshape(self.data[k].shape)
                .transpose(1, 0, 2, 3, 4)
            )  # [time, num_sim, level, lat, lon]

        self.time_steps = self.data[self.include_keys[0]].shape[0]

        # ----------- contruct ground truth param table ------------
        param_samples = {
            k: np.linspace(v.min_, v.max_, self.grid_size[i]) for i, (k, v) in enumerate(self.param_spaces.items())
        }
        PARAM_GRID = np.stack(np.meshgrid(*[list(v) for v in param_samples.values()], indexing="ij"), axis=-1).reshape(
            -1, len(param_samples)
        )
        param_sample_values = list(param_samples.values())
        self.params = np.stack(
            [np.searchsorted(param_sample_values[i], PARAM_GRID[:, i]) for i in range(PARAM_GRID.shape[-1])], -1
        )

    def __len__(self):
        return self.num_simulations * self.level_dim * self.lat_dim * self.lon_dim

    def __sample_location__(self):
        """
        Sample a location from the grid.
        """
        sampled_lev = np.random.randint(self.level_dim)
        sampled_lat = np.random.randint(self.lat_dim)
        sampled_lon = np.random.randint(self.lon_dim)
        return (sampled_lev, sampled_lat, sampled_lon)

    def __sample_simulation__(self, simulation_index, shared_ids):
        """
        Sample a simulation index from the dataset
        """
        multi_dim_index = np.unravel_index(simulation_index, self.grid_size)
        aug_multi_dim_index = [
            np.random.choice(np.delete(np.arange(self.grid_size[i]), multi_dim_index[i]))
            if i not in shared_ids
            else multi_dim_index[i]
            for i in range(len(self.grid_size))
        ]
        return np.ravel_multi_index(aug_multi_dim_index, self.grid_size)

    def __retrieve_item__(self, simulation_index: int, location: Tuple[int]):
        traj = np.stack(
            [self.data[k][:, simulation_index, *location] for k in self.include_keys],
            axis=-1,
        )
        param = {k: self.params[simulation_index][i] for i, k in enumerate(self.param_spaces.keys())}
        return simulation_index, location, param, traj

    def __getitem__(self, index: int):
        multi_dim_ind = np.unravel_index(index, (self.num_simulations, self.level_dim, self.lat_dim, self.lon_dim))
        simulation_index = multi_dim_ind[0]
        _, _, params, trajectory = self.__retrieve_item__(simulation_index, multi_dim_ind[1:])
        if self.mode == "train":
            # sample chrunks at random position in the trajectory
            if self.chunk_size < self.time_steps // 2:
                # randomly sample the time index
                time_index = np.random.randint(self.time_steps - 2 * self.chunk_size)
            else:
                # return the whole trajectory
                time_index = 0
            return {
                "index": simulation_index,
                "time_index": time_index,
                "location": multi_dim_ind[1:],
                "gt_params": params,  # n_views, batch_size, 4
                "states": trajectory[time_index : time_index + self.chunk_size],
            }
        else:
            return {
                "index": simulation_index,
                "time_index": self.time_steps - self.chunk_size,
                "location": multi_dim_ind[1:],
                "gt_params": params,  # n_views, batch_size, 4
                "states": trajectory[-self.chunk_size :],
            }  # index of the location (i,j) and (i',j'

    def __get_augmented_view__(
        self, simulation_index, location, shared_ids, aug_location=True, aug_simulation=True
    ) -> Tuple[int, Tuple]:
        # sample the augmented trajectory
        assert aug_location or aug_simulation, "change at least one of the options"
        if aug_simulation:
            sampled_simulation_index = self.__sample_simulation__(simulation_index, shared_ids)
        else:
            sampled_simulation_index = simulation_index

        if aug_location:
            sampled_location = self.__sample_location__()
        else:
            sampled_location = location
        return sampled_simulation_index, sampled_location

    def abstract_collate_fn(self, batch, shared_ids):
        if len(self.param_spaces) == 1:
            # only one factor of variation
            n_views = 3
            args = [(False, True), (True, True)]
        else:
            n_views = 4
            args = [(True, False), (False, True), (True, True)]
        simulation_ids = [[] for _ in range(n_views)]
        locations = [[] for _ in range(n_views)]
        states = [[] for _ in range(n_views)]
        params = {k: [[] for _ in range(self.n_views)] for k in self.param_spaces.keys()}

        # batch: list of dictionary
        for b in batch:
            simulation_index, location = b["index"], b["location"]
            simulation_ids[0] += [simulation_index]
            locations[0] += [location]
            states[0] += [b["states"]]
            for k in params:
                params[k][0] += [b["gt_params"][k]]

            for i, settings in enumerate(args):
                sampled_simulation_index, sampled_location, sampled_params, sampled_traj = self.__retrieve_item__(
                    *self.__get_augmented_view__(simulation_index, location, shared_ids, *settings)
                )
                simulation_ids[i + 1] += [sampled_simulation_index]
                locations[i + 1] += [sampled_location]
                states[i + 1] += [sampled_traj]
                for k in params:
                    params[k][i + 1] += [sampled_params[k]]

        for k in params:
            params[k] = np.stack(params[k])

        batch_dict = {
            "shared_index": self.shared_ids,
            "index": simulation_ids,  # n_views, batch_size # simulation index
            "location": locations,  # n_views, batch_size, 3
            "gt_params": params,  # n_views, batch_size, 3
            "states": torch.from_numpy(np.stack(states)),  # n_views, batch_size, time, 3
        }
        return batch_dict

    def default_collate_fn(self, batch):
        return self.abstract_collate_fn(batch, self.shared_ids)

    def random_collate_fn(self, batch):
        # TODDO: double check if the shared ids will be randomly sampled every time
        return self.abstract_collate_fn(batch, np.random.choice(len(self.param_spaces), 1, replace=False))

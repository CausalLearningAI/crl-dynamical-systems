import random
from typing import List

import metrics
import numpy as np
import torch
from lightning.pytorch import LightningModule
from torch import nn
from utils.dct import LinearDCT
from utils.misc import feature_sharing_fn


class AbstractIdentifier(LightningModule):
    def __init__(
        self,
        state_dim: int = 1,
        n_steps: int = 60,
        n_iv_steps: int = 10,
        n_views: int = 2,
        hidden_dim: int = 1024,
        param_dim=20,
        dct_layer: bool = False,
        freq_frac_to_keep: float = 0.5,
        learning_rate: torch.float64 = 1e-5,
        eval_metrics: List[str] = [],
        factor_type="discrete",
        **kwargs,
    ):
        super().__init__()
        # ------------- define dimensionality -----------------
        self.state_dim = state_dim
        self.n_steps = n_steps
        self.n_iv_steps = n_iv_steps
        self.n_views = n_views
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim
        self.dct_layer = dct_layer
        self.freq_frac_to_keep = freq_frac_to_keep
        self.learning_rate = learning_rate
        self.eval_metrics = eval_metrics
        self.factor_type = factor_type

        # ------- construct code sharing map ------------
        assert n_views <= 3, "only support up to 3 views"
        splits = [tuple(s) for s in np.array_split(np.arange(param_dim), n_views)]
        subsets = [(0, 1), (0, 2), (0, 1, 2)]
        if n_views == 2:
            # for sst
            self.code_sharing = {s: subsets[i] for i, s in enumerate(splits[:1])}
        if n_views in [1, 3]:  # TODO: double check here
            # for speedy weather
            self.code_sharing = {s: subsets[i] for i, s in enumerate(splits)}
        # code_sharing = {
        #     splits[0]: (0, 1),  # share layer thickness
        #     splits[1]: (0, 2),  # share local features
        #     splits[2]: (0, 1, 2),  # share other global features
        # }  # subset of views: shared coding dims
        self.misc = {
            "pred_params": [],
            "pred_states": [],
            "gbt": [],
            "gt_params": [],
            "id_score_linear": [],
            "id_score_nonlinear": [],
        }

        if self.code_sharing is None:
            print("Code sharing map is None, latents will not be averaged, make sure this is intended.")
            if "identifiability" in self.eval_metrics:
                self.eval_metrics.remove("identifiability")
            self.shared_encoding = [range(self.param_dim)]
        else:
            self.shared_encoding = list(self.code_sharing.keys())

        # ------------- add dct transform when specified ----------
        if dct_layer:
            self.dct: nn.Module = LinearDCT(self.n_steps, "dct", norm="ortho")
            self.idct: nn.Module = LinearDCT(self.n_steps, "idct", norm="ortho")
            self.input_dim = int(self.freq_frac_to_keep * self.n_steps) * self.state_dim
        else:
            self.input_dim = self.state_dim * self.n_steps

        self.data_type = torch.float32

    def state_transform(self, states: torch.Tensor):
        # states: [n_views, bs, n_step, state_dim]
        freqs: torch.Tensor = self.dct(states.swapaxes(-1, -2)).swapaxes(-1, -2)
        return freqs[..., : int(self.freq_frac_to_keep * self.n_steps), :]

    def state_inverse_transform(self, freqs: torch.Tensor):
        # freqs: [bs, n_freqs_to_keep, state_dim]
        # fill the high-frequency that we droped before with zero
        freqs: torch.Tensor = torch.cat(
            [freqs, torch.zeros(*freqs.shape[:2], self.n_steps - freqs.shape[-2], freqs.shape[-1]).type_as(freqs)],
            dim=-2,
        )
        return self.idct(freqs.swapaxes(-1, -2)).swapaxes(-1, -2)

    def feature_sharing(self, params: torch.Tensor, **kwargs):
        # this should be inherent to the data generating process, so it should be an attribute
        # to the corresponding dataset
        # only one view, no sharing
        if len(params) == 1:
            return params
        else:
            return feature_sharing_fn(params, n_views=self.n_views, code_sharing=self.code_sharing, **kwargs)

    def __init__encoder__(self):
        raise NotImplementedError

    def __init_solver__(self):
        raise NotImplementedError

    def forward(self, states: torch.Tensor):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        states = batch["states"].to(self.data_type)
        if self.dct_layer:
            states = self.state_transform(states)
        params_hat = (
            self.encoder(states.reshape(-1, states.shape[-2] * states.shape[-1]))
            .reshape(*states.shape[:-2], -1)
            .cpu()
            .numpy()
        )
        # store predicted parameters for the whole dataset (earth)
        self.misc["pred_params"].append(params_hat)
        if "gt_params" in batch:
            self.misc["gt_params"].append(torch.stack(list(batch["gt_params"].values()), -1).cpu().numpy())

    def forecasting_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        if "identifiability" in self.eval_metrics:
            self.validation_step(batch, batch_idx)
        if "forecasting" in self.eval_metrics:
            self.forecasting_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        if "identifiability" not in self.eval_metrics:
            pass
        else:
            pred_params = np.concatenate(self.misc["pred_params"], 0).squeeze()  # (ds, param_dim)
            gt_params = np.concatenate(self.misc["gt_params"], 0)  # (ds, param_dim)
            X = pred_params  # [..., :4]
            if self.factor_type == "discrete":
                y = gt_params.astype("int8")  # bs, 4
            else:
                y = gt_params

            # shuffle validation data
            zipped = list(zip(X, y))
            random.shuffle(zipped)
            X, y = zip(*zipped)

            X, y = np.asanyarray(X), np.asanyarray(y)
            # compute accuracy score for the shared latents
            # we had the first partition to encode the sharing part
            Xs: List[np.ndarray] = [X[:, self.shared_encoding[0]]]  # [X[:, s] for s in self.shared_encoding]
            id_score_linear, id_score_nonlinear = metrics._compute_identifiability_score(
                Xs,
                y.T,
                factor_types=[self.factor_type] * y.shape[-1],
            )
            self.misc["id_score_linear"].append(id_score_linear)
            self.misc["id_score_nonlinear"].append(id_score_nonlinear)
            self.log("id_score_linear", round(id_score_linear.max(-1).mean(), 4), on_epoch=True, batch_size=X.shape[0])
            self.log(
                "id_score_nonlinear", round(id_score_nonlinear.max(-1).mean(), 4), on_epoch=True, batch_size=X.shape[0]
            )

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

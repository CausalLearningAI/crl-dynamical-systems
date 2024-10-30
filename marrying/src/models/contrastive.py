from typing import List

import torch
from models.abstract import AbstractIdentifier
from utils.losses import infonce_loss
from utils.misc import xavier_init
from utils.mlp import MLP


class ContrastiveIdentifier(AbstractIdentifier):
    def __init__(
        self,
        state_dim: int = 1,
        n_steps: int = 60,
        n_iv_steps: int = 10,
        n_views: int = 2,
        hidden_dim: int = 1024,
        param_dim=20,
        dct_layer: bool = False,
        freq_frac_to_keep=0.5,
        learning_rate: torch.float64 = 1e-5,
        eval_metrics: List[str] = [],
        factor_type="discrete",
        # contrastive identifier-specific hyperparameters
        tau: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            state_dim=state_dim,
            n_steps=n_steps,
            n_iv_steps=n_iv_steps,
            n_views=n_views,
            hidden_dim=hidden_dim,
            param_dim=param_dim,
            dct_layer=dct_layer,
            freq_frac_to_keep=freq_frac_to_keep,
            learning_rate=learning_rate,
            eval_metrics=eval_metrics,
            factor_type=factor_type,
        )

        # ------------ contrastive learning specific hyper parameters -----------
        self.tau = tau

        # ------------initialize parameter encoder -----------------
        self.__init__encoder__()
        # initialize weights
        if self.train():
            xavier_init(self.encoder)

        # ------------ define CL loss ------------
        self.sim_metric = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def __init__encoder__(self):
        self.encoder = MLP(self.input_dim, self.param_dim, self.hidden_dim)

    def loss(self, z_rec_tuple: torch.Tensor):
        return infonce_loss(
            z_rec_tuple,
            sim_metric=self.sim_metric,
            criterion=self.criterion,
            tau=self.tau,
            projector=(lambda x: x),
            estimated_content_indices=list(self.code_sharing.keys()),
            subsets=list(self.code_sharing.values()),
        )

    def forward(self, states: torch.Tensor, **kwargs):
        if self.dct_layer:
            states: torch.Tensor = self.state_transform(states)
        # states: [n_views, bs, ts, state_dim]
        return self.encoder(states.reshape(-1, states.shape[-2] * states.shape[-1]))

    def training_step(self, batch, batch_idx):
        states = batch["states"].float()
        params_hat = self.forward(states).reshape(states.shape[0], states.shape[1], -1)
        loss = self.loss(params_hat)
        self.log("CL_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def forecasting_step(self, batch, batch_id):
        raise NotImplementedError("Forecasting is not available for contrastive identifier.")

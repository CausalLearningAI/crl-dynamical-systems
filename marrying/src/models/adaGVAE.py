from typing import List

import torch
from models.abstract import AbstractIdentifier
from torch.distributions import Normal
from utils.mlp import MLP


class AdaGVAE(AbstractIdentifier):
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
        # ada-GVAEspeciifc hyperparameters
        device="cpu",
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
            **kwargs,
        )

        # AdaGVAE-specific settings
        assert len(self.code_sharing) == 1, "only one augmented view"
        self.loss = torch.nn.MSELoss()
        self.prior = Normal(torch.zeros(param_dim).to(device), torch.ones(param_dim).to(device))
        self.shared_dims = list(self.code_sharing.keys())[0]

        self.__init__encoder__()
        self.__init_solver__()

        self.save_hyperparameters()

    def __init__encoder__(self):
        self.encoder = MLP(
            input_dim=self.input_dim, output_dim=self.param_dim, hidden_dim=self.hidden_dim, num_layers=5
        )
        self.mean_head = MLP(
            input_dim=self.param_dim, output_dim=self.param_dim, hidden_dim=self.hidden_dim, num_layers=2
        )

        self.logvar_head = MLP(
            input_dim=self.param_dim, output_dim=self.param_dim, hidden_dim=self.hidden_dim, num_layers=2
        )

    def __init_solver__(self):
        self.decoder = MLP(
            input_dim=self.param_dim + self.n_iv_steps * self.state_dim,
            output_dim=self.state_dim * self.n_steps,
            hidden_dim=self.hidden_dim,
            num_layers=5,
        )

    def encode(self, states: torch.Tensor):
        if self.dct_layer:
            states: torch.Tensor = self.state_transform(states)
        params = self.encoder(states.reshape(-1, states.shape[-2] * states.shape[-1])).reshape(
            states.shape[0], -1, self.param_dim
        )  # (n_views, batch_size, param_dim)
        means = self.mean_head(params)
        logvars = self.logvar_head(params)
        scales = torch.exp(logvars / 2)
        if len(params) == 1:
            return [torch.distributions.Normal(means, scales)]
        else:
            shared = means.clone()
            avg = (means[0, :, self.shared_dims[0]] + means[1, :, self.shared_dims[0]]) / 2
            shared[:, :, self.shared_dims[0]] = avg.expand_as(shared[:, :, self.shared_dims[0]])
            scales = torch.exp(logvars / 2)
            return [Normal(shared[0], scales[0]), Normal(shared[1], scales[1])]

    def decode(self, posteriors: torch.distributions.Distribution, states: torch.Tensor):
        n_views, batch_size, _, _ = states.shape
        if len(posteriors) > 1:  # training mode
            zs = torch.stack([p.rsample() for p in posteriors], 0)  # shape: [n_views, batch_size, param_dim]
        else:
            zs = posteriors[0].mean  # shape: [n_views=1, batch_size, param_dim]
        ivps = states[..., : self.n_iv_steps, :]  # shape: [n_views, batch_size, n_iv_steps, state_dim]
        latents = torch.cat([zs, ivps.reshape(zs.shape[0], zs.shape[1], -1)], -1)
        return self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(n_views, batch_size, -1, self.state_dim)

    def forward(self, states: torch.Tensor):
        posteriors = self.encode(states)
        u0 = self.decode(posteriors, states)
        return u0.reshape(*states.shape), posteriors

    def training_step(self, batch, batch_idx):
        states = batch["states"].float().reshape(self.n_views, -1, self.n_steps, self.state_dim)
        u0, posteriors = self.forward(states)
        recon = self.loss(u0, states)
        D_kl = torch.stack([torch.distributions.kl.kl_divergence(p, self.prior) for p in posteriors]).sum()
        loss = recon + D_kl
        self.log("recon_loss", recon, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("kl_loss", D_kl, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        states = batch["states"].float().reshape(1, -1, self.n_steps, self.state_dim)
        u0, posteriors = self.forward(states)
        recon = self.loss(u0, states)
        self.misc["pred_params"].append(posteriors[0].mean.cpu().numpy().squeeze())
        if "gt_params" in batch:
            self.misc["gt_params"].append(torch.stack(list(batch["gt_params"].values()), -1).cpu().numpy())
        self.log("recon_loss", recon, on_epoch=True, prog_bar=True, logger=True)

    def forecasting_step(self, batch, batch_id):
        # n_views = 1 in evaluation
        # select the first half as encoder input
        # states takes two chunks; use the first half as input to the encoder;
        # then use \hat{\theta} to predict the second half
        states = batch["states"].float().reshape(1, -1, 2 * self.n_steps, self.state_dim)
        input_states = states[..., : self.n_steps, :]
        future_states = states[..., self.n_steps :, :]

        posteriors = self.encode(input_states)
        zs = posteriors[0].mean  # shape: [n_views=1, batch_size, param_dim]
        ivps = future_states[..., : self.n_iv_steps, :]  # shape: [n_views, batch_size, n_iv_steps, state_dim]
        latents = torch.cat([zs, ivps.reshape(zs.shape[0], zs.shape[1], -1)], -1)
        u0s = self.decoder(latents.reshape(-1, latents.shape[-1])).reshape(*future_states.shape)
        forecast_loss = self.loss(u0s, future_states)
        self.log("forecast_loss", forecast_loss, on_epoch=True, prog_bar=True)

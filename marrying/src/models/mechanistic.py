from typing import List

import torch
import torch.nn as nn
from models import AbstractIdentifier
from scipy.special import logit
from solver.ode_layer import ODESYSLayer
from utils.mlp import MLP


class MechanisticIdentifier(AbstractIdentifier):
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
        # MNN-specific hyperparameters
        batch_size: int = 10,
        order: int = 2,  # order of polynomial of MNN
        # Multiview-speciifc hyperparameters
        alignment_reg=10,
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

        # ------------- define MNN-specific hyper-parameters ------------
        # number of discretized time steps of the observation
        # the order of derivatives we use to approximate,
        # not necessarily the ground truth order of the ODE (which is unknown)
        self.batch_size = batch_size
        self.order = order
        self.alignment_reg = alignment_reg

        # ------------- initialize parameter encoder -----------------
        self.__init__encoder__()

        # --------------  initialize mechanistic layer ---------------
        self.__init__solver__()

        # --------------  define loss ---------------
        self.loss = nn.MSELoss().double()
        self.data_type = torch.float64
        self.to(self.data_type)

        # -----------log hyperparameters through lightning ----------
        self.save_hyperparameters()

    def __init__encoder__(self):
        self.encoder = MLP(self.input_dim, self.param_dim, self.hidden_dim)

    def __init__solver__(self):
        # define a MNN layer
        self.ode_layer = ODESYSLayer(
            bs=self.batch_size * self.n_views,
            n_ind_dim=1,
            order=self.order,
            n_equations=self.state_dim,  # equals number of states
            n_dim=self.state_dim,
            n_iv=1,
            n_step=self.n_steps,
            n_iv_steps=self.n_iv_steps,
            solver_dbl=True,
        )

        self.n_coeff = self.n_steps * (self.order + 1)
        self.step_dim = (self.n_steps - 1) * self.state_dim
        # define the dimensions
        self.rhs_dim = self.state_dim * self.n_steps  # time_steps * state_dim
        self.coeff_dim = (
            self.ode_layer.n_ind_dim
            * self.ode_layer.n_equations
            * self.ode_layer.n_step
            * self.ode_layer.n_dim
            * (self.order + 1)
        )
        # decode from params to rhs
        self.rhs_t = MLP(input_dim=self.param_dim, output_dim=self.rhs_dim, hidden_dim=self.hidden_dim, num_layers=3)

        self.coeffs_mlp = MLP(
            input_dim=self.param_dim,
            output_dim=self.coeff_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
        )

        self.pre_steps_mlp = nn.Sequential(
            nn.Linear(self.param_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )

        self.steps_layer = nn.Linear(self.hidden_dim, self.step_dim)
        # set step bias to make initial step 0.1
        step_bias = logit(0.1)
        self.steps_layer.weight.data.fill_(0.0)
        self.steps_layer.bias.data.fill_(step_bias)

    def decode_from_params(self, params: torch.Tensor):
        # compute righthandside of the ODE
        rhs: torch.Tensor = self.rhs_t(params)  # (bs, n_step*state_dim)
        # time varying ODE coefficients
        coeffs: torch.Tensor = self.coeffs_mlp(params)  # (bs, n_step*state_dim*(order+1))
        # learned steps
        _steps = self.pre_steps_mlp(params)  # (bs, hidden_dim)
        steps: torch.Tensor = self.steps_layer(_steps)  # (bs, n_steps-1)
        steps: torch.Tensor = torch.sigmoid(steps).clip(min=0.001, max=0.999)  # (bs, n_steps-1)
        return rhs, coeffs, steps

    def solve(self, params: torch.Tensor, iv_rhs: torch.Tensor):
        rhs, coeffs, steps = self.decode_from_params(params)
        u0, u1, u2, eps, steps = self.ode_layer(coeffs=coeffs, rhs=rhs, iv_rhs=iv_rhs, steps=steps)
        u0 = u0.squeeze(1)  # (n_views*bs, ts, state_dim)
        return u0.reshape(self.n_views, -1, self.n_steps, self.state_dim)

    def forward(self, states: torch.Tensor, **kwargs):
        # states: (bs, n_step, state_dim)
        # extarct iv steps before dct layer, make sure it is in the time domain
        iv_rhs = states[..., : self.n_iv_steps, :]  # (bs, n_iv_steps, state_dim)
        if self.dct_layer:
            states: torch.Tensor = self.state_transform(states)
        # parameter encoding
        params: torch.Tensor = self.encoder(states.reshape(-1, states.shape[-2] * states.shape[-1]))  # (bs, param_dim)
        params: torch.Tensor = params.reshape(-1, self.batch_size, self.param_dim)  # (n_views, bs, param_dim)
        iv_rhs = iv_rhs.reshape(
            -1, self.batch_size, self.n_iv_steps, self.state_dim
        )  # (n_views, bs, n_iv_steps, state_dim)
        shared = self.feature_sharing(params, **kwargs)
        # no matter apply dct layer or not, u0 always in time domain
        # shape: [n_views, bs, ts, state_dim]
        u0s = self.solve(shared.view(-1, self.param_dim), iv_rhs.view(-1, self.n_iv_steps, self.state_dim))
        if self.dct_layer:
            u0s = self.state_transform(u0s.double())  # to convert u0s to the freq domain; make sure it is double
        return states, u0s, params, shared  # u0: [bs, ts, state_dim], params: [bs, param_dim]

    def training_step(self, batch, batch_idx):
        # [n_views, bs, ts, state_dim]
        batch["states"] = batch["states"].to(self.data_type)
        # depending on if we have dct layer or not, the output states could be in freq space
        states, u0s, params, shared = self.forward(
            **batch
        )  # here: [n_views * bs, ts, state_dim], [n_views * bs, param_dim]
        # states = states.reshape(-1, self.n_step, self.state_dim)
        # u0: [n_views, bs, ts, state_dim], params: [n_views, bs, param_dim]
        recon_loss = self.loss(u0s.double().reshape(*states.shape), states.double())
        if self.alignment_reg > 0.0:
            param_loss = self.loss(params[..., : self.n_views - 1], shared[..., : self.n_views - 1])
            loss = recon_loss + self.alignment_reg * param_loss  # TODO: maybe use another optimizer later
            self.log("param_loss", param_loss, prog_bar=True, on_step=True, on_epoch=True)
        else:
            loss = recon_loss
        self.log("train_loss", recon_loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        states = batch["states"].to(self.data_type)
        if self.dct_layer:
            states = self.state_transform(states)
        params = self.encoder(states.reshape(-1, states.shape[-2] * states.shape[-1])).cpu().numpy()
        self.misc["pred_params"].append(params)
        if "gt_params" in batch:
            self.misc["gt_params"].append(torch.stack(list(batch["gt_params"].values()), -1).cpu().numpy())

    def forecasting_step(self, batch, batch_id):
        # select the first half as encoder input
        states = batch["states"].to(self.data_type)
        input_states = states[..., : self.n_steps, :]
        future_states = states[..., self.n_steps :, :]
        if self.dct_layer:
            input_states = self.state_transform(input_states)

        params = self.encoder(input_states.reshape(-1, input_states.shape[-2] * input_states.shape[-1]))

        iv_rhs = future_states[..., : self.n_iv_steps, :].reshape(
            -1, self.batch_size, self.n_iv_steps, self.state_dim
        )  # (n_views, bs, n_iv_steps, state_dim)
        # no matter apply dct layer or not, u0 always in time domain
        # shape: [n_views, bs, ts, state_dim]
        u0s = self.solve(params.view(-1, self.param_dim), iv_rhs.view(-1, self.n_iv_steps, self.state_dim))
        u0s = u0s.reshape(*future_states.shape)
        forecast_loss = self.loss(u0s.double(), future_states.double())
        self.log("forecast_loss", forecast_loss, on_epoch=True, prog_bar=True)

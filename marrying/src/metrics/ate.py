import numpy as np

# from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import torch
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.linear_model import LinearRegression, LogisticRegression


def _compute_ate(
    Y, T, X, method="aipw", model_propensity=LogisticRegression(), model_outcome=None, T_control=1, T_treatment=2
):
    if model_outcome is None:
        if Y.dtype == torch.int32 or Y.dtype == torch.int64:
            model_outcome = LogisticRegression()
        else:
            model_outcome = LinearRegression()

    if method == "ead":
        model = EAD()
    elif method == "aipw":
        model = AIPW(model_propensity=model_propensity, model_outcome=model_outcome)
    elif method == "slearner":
        model = SLearner(overall_model=model_outcome)
    elif method == "tlearner":
        model = TLearner(models=model_outcome)
    elif method == "xlearner":
        model = XLearner(
            models=model_outcome,
            propensity_model=model_propensity,
            # cate_models = model_outcome
        )
    elif method == "drlearner":
        model = DRLearner(
            discrete_outcome=False,
            # model_propensity = model_propensity,
            # model_regression = model_outcome,
            # model_final = LinearRegression()
        )
    elif method == "causalforest":
        model = CausalForestDML(
            discrete_outcome=False,
            discrete_treatment=True,
            # model_t = model_propensity,
            # model_y = model_outcome
        )
    else:
        raise ValueError(f"'{method}' method for ATE estimation not implemented.")
    Y = Y[(T == T_control) | (T == T_treatment)]
    X = X[(T == T_control) | (T == T_treatment)]
    T = T[(T == T_control) | (T == T_treatment)].int()
    T[T == T_control] = 0
    T[T == T_treatment] = 1
    model.fit(Y=Y, T=T, X=X)
    ate = model.effect(X).mean()
    return ate


class EAD:
    def __init__(self):
        self.name = "EAD"

    def fit(self, Y, T, X):
        E_Y_control = Y[T == 0].sum() / len(Y[T == 0])
        E_Y_treated = Y[T == 1].sum() / len(Y[T == 1])
        ead = E_Y_treated - E_Y_control
        self.ate = ead.item()

    def effect(self, _):
        return np.array(self.ate)


class AIPW:
    def __init__(self, model_propensity, model_outcome):
        self.name = "AIPW"
        self.model_propensity = model_propensity
        self.model_outcome = model_outcome

    def fit(self, Y, T, X):
        N = len(Y)
        self.model_propensity.fit(X=X, y=T)
        self.model_outcome.fit(X=torch.cat((X, T.reshape(N, 1)), dim=1), y=Y)
        mu0 = self.model_outcome.predict(torch.cat((X, torch.zeros(N, 1)), dim=1))
        mu1 = self.model_outcome.predict(torch.cat((X, torch.ones(N, 1)), dim=1))
        ps = self.model_propensity.predict_proba(X)[:, 1]
        self.ite = mu1 - mu0 + T.numpy() * (Y.numpy() - mu1) / (ps) - (1 - T.numpy()) * (Y.numpy() - mu0) / (1 - ps)

    # self.ate = ite.mean().item()

    def effect(self, _):
        return self.ite


if __name__ == "__main__":
    Y = np.load("/nfs/scistore19/locatgrp/dyao/ode_discovery/zonal_avg.npy")
    from scipy.io import netcdf

    data_path = "/nfs/scistore19/locatgrp/dyao/DATA/sst/sst.wkmean.1990-present.nc"
    sst_netcdf = netcdf.NetCDFFile(data_path, "r")
    keys = list(sst_netcdf.variables.keys())
    data = {k: np.asarray(sst_netcdf.variables[k][:].byteswap().newbyteorder()) for k in keys}

    Y = data["sst"].reshape(-1, 180, 360)[: 52 * 4].mean((0, 2))

    X = np.load("/nfs/scistore19/locatgrp/dyao/ode_discovery/pred_params.npy")
    X = X.reshape(180, 360, -1).mean(1)
    T = np.load("/nfs/scistore19/locatgrp/dyao/ode_discovery/climate_zone_labels.npy")
    T = T[::360]

    method = "aipw"

    X, Y, T = torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(T).int()
    ate_1990 = _compute_ate(
        Y, T, X, method=method, model_propensity=LogisticRegression(), model_outcome=None, T_control=0, T_treatment=2
    )

    Y = np.load("/nfs/scistore19/locatgrp/dyao/ode_discovery/zonal_avg_recent.npy")
    X = np.load("/nfs/scistore19/locatgrp/dyao/ode_discovery/pred_params_recent.npy")

    X = X.reshape(180, 360, -1).mean(1)
    X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    ate_2016 = _compute_ate(
        Y, T, X, method=method, model_propensity=LogisticRegression(), model_outcome=None, T_control=0, T_treatment=2
    )
    print(ate_1990, ate_2016, np.abs((ate_2016 - ate_1990) / ate_1990))

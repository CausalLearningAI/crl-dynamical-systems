import math
from collections import Counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


def feature_sharing_fn(encodings: torch.Tensor, n_views: int, code_sharing: Dict[int, List[int]] = None, **kwargs):
    # code sharing: {codes, List[views]}
    if n_views == 1 or code_sharing is None:  # no sharing
        return encodings
    else:
        # TODO: add avg
        # if code_sharing is None:
        #     code_sharing = {[i]: [0, 1] for i in kwargs["shared_index"]}
        shared = encodings.clone()
        for code_indices, views in code_sharing.items():
            if isinstance(code_indices, int):
                code_indices = [code_indices]
            for code in code_indices:
                mean_per_code = encodings[views, ..., code].mean(0)  # (views_sharing_factor, bs, 1)
                shared[views, ..., code] = mean_per_code.expand(shared[views, ..., code].shape)
        return shared


def xavier_init(model):
    for name, param in model.named_parameters():
        if len(param.shape) < 2:
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


def calculated_torch_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f}MB".format(size_all_mb))


def evaluate_prediction(regression_model, metric, X_train, y_train, X_test, y_test):
    # handle edge cases when inputs or labels are zero-dimensional
    if any([0 in x.shape for x in [X_train, y_train, X_test, y_test]]):
        return np.nan
    assert X_train.shape[1] == X_test.shape[1]
    # assert y_train.shape[1] == y_test.shape[1]
    # handle edge cases when the inputs are one-dimensional
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)
    return metric(y_test, y_pred)


def generate_batch_factor_code(ground_truth_data, representation_function, num_points, random_state, batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_function: Function that takes observation as input and
        outputs a representation.
      num_points: Number of points to sample.
      random_state: Numpy random state used for randomness.
      batch_size: Batchsize to sample points.

    Returns:
      representations: Codes (num_codes, num_points)-np array.
      factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


# for discovery


def StringConvolve(l1, l2):
    res = []
    for i in range(len(l1)):
        for j in range(min(i, len(l2) - 1), len(l2)):
            if l1[i] == "1":
                res.append(l2[j])
            else:
                res.append(l1[i] + l2[j])
    return res


def StringMultFormat(str):
    sortedStr = sorted(str)
    availablesVars = list(set(sortedStr))
    counts = Counter(sortedStr)

    rs = ""

    for i in range(len(availablesVars)):
        power = counts[availablesVars[i]]
        if power == 1:
            rs = rs + availablesVars[i] + " "
        else:
            rs = rs + availablesVars[i] + "^{} ".format(power)
    return rs


def StringTerms(StateVariables, _polyorder):
    terms = []
    s1 = ["1"]
    for i in range(_polyorder):
        terms += StringConvolve(s1, StateVariables)
        s1 = terms
    return ["1"] + terms


def StringModelView(_xi, _polyorder=2, StateVariables=[], threshold=1e-3):
    terms = []
    s1 = ["1"]
    for i in range(_polyorder):
        terms += StringConvolve(s1, StateVariables)
        s1 = terms
    terms = ["1"] + terms
    for i in range(len(StateVariables)):
        if abs(_xi[0, i]) > threshold:
            row = "d" + StateVariables[i] + "/dt = " + "{: 2.3f}".format(_xi[0, i])
            sp = " + "
        else:
            row = "d" + StateVariables[i] + "/dt = "
            sp = ""

        for j in range(1, len(_xi)):
            # ss = "{: 2.2f}".format(_xi[j, i])
            if abs(_xi[j, i]) > threshold:
                row = row + sp + "{: 2.3f}".format(_xi[j, i]) + " " + StringMultFormat(terms[j])
                sp = " + "
        print(row)


# read speedy weather parameters
def parse_text_to_nested_dict(text_file):
    with open(text_file, "r") as file:
        text = file.read()
    nested_dict = {}
    current_dict = nested_dict

    for line in text.split("\n"):
        line = line.strip()
        if line:
            if "::" not in line and "<:" not in line:
                continue
            elif line.startswith("├") or line.startswith("└"):
                key_value = line[2:].strip().split(" = ")
                key = key_value[0].split("::")[0]
                value = key_value[1]
                current_dict[key] = value
            elif line.startswith(" "):
                key_value = line.strip().split(" = ")
                key = key_value[0]
                value = key_value[1]
                current_dict[key] = value
            else:
                current_dict = nested_dict

    nested_dict["mu_virt_temp"] = nested_dict.pop("μ_virt_temp")
    nested_dict["kappa"] = nested_dict.pop("κ")
    return nested_dict


def plot_gbt(gbt: np.ndarray, file_name="gbt_discrete.png"):
    # Plot the results
    fig, ax = plt.subplots(figsize=(7, 5))
    matrix = np.stack(gbt).mean(0).T
    # matrix[matrix < 0.05] = 0
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")
    fig.colorbar(im, ax=ax)
    plt.xlabel(r"$\hat{\theta}$")
    plt.ylabel(r"$\theta$")
    plt.savefig(file_name, bbox_inches="tight")


# def feature_sharing_fn(params: torch.Tensor, n_views: int, shared_dims: List[List[int]], **kwargs):
#     if n_views == 1:
#         return params
#     else:
#         assert len(shared_dims) == n_views - 1, "shared_dims must be a list of length n_views-1"
#         # pass batch to kwargs
#         # params: [n_views, bs, param_dim] predicted params; learning encoding
#         shared = [[] for _ in range(params.shape[-1])]

#         for i, ids in zip(range(1, n_views), ids):
#             vals=[]
#             base_params = params[0, ..., shared_dims[i - 1]]  # [batch_size, shared_dims[i-1]
#             view_params = params[i, ..., shared_dims[i - 1]]  # [batch_size, shared_dims[i-1]
#             shared_param = torch.stack([base_params, view_params], 0).mean(0)
#             shared[0, ..., shared_dims[i - 1]] = shared_param
#             shared[i, ..., shared_dims[i - 1]] = shared_param
#         # for i in range(1, self.n_views):
#         #     shared_param = params[[0, i], ..., i-1].mean(0)
#         #     shared[[0, i], ..., i-1] = shared_param.expand_as(params[[0, i], :, i-1])
#         return shared

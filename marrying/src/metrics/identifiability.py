from typing import List

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor


def evaluate_prediction(model, metric, X_train, y_train, X_test, y_test):
    """
    Evaluates the performance of a model by fitting it on training data, predicting on test data,
    and calculating a specified metric between the predicted and true labels.

    Parameters:
        model (object): The machine learning model to be evaluated.
        metric (function): The evaluation metric to be used.
        X_train (array-like): The training input samples.
        y_train (array-like): The training target values.
        X_test (array-like): The test input samples.
        y_test (array-like): The test target values.

    Returns:
        float: The evaluation score calculated using the specified metric.
    """
    # handle edge cases when inputs or labels are zero-dimensional
    if any([0 in x.shape for x in [X_train, y_train, X_test, y_test]]):
        return np.nan
    assert X_train.shape[1] == X_test.shape[1]
    if y_train.ndim > 1:
        assert y_train.shape[1] == y_test.shape[1]
        if y_train.shape[1] == 1:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
    # handle edge cases when the inputs are one-dimensional
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric(y_test, y_pred)


def _compute_identifiability_score(
    Xs: List[np.ndarray], ys: List[np.ndarray], factor_types: List[str], grid_search_eval: bool = False
):
    # compute R2 score / acc for linear and nonlinear models
    # for each chunk of data, we predict the factors individually

    # check if factor ix is discrete for modality m
    # for continuous factors, do regression and compute R2 score
    # for discrete factors, do classification and compute accuracy
    id_score_linear = []
    id_score_nonlinear = []
    for i, y in enumerate(ys):
        for X in Xs:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
            data = (X_train, y_train, X_test, y_test)
            linear_res = []
            nonlinear_res = []

            factor_type = factor_types[i]
            assert factor_type in ["continuous", "discrete"], "invalid factor type"
            if factor_type == "continuous":
                # linear regression
                linreg = LinearRegression(n_jobs=-1)
                linear = evaluate_prediction(linreg, r2_score, *data)
                if grid_search_eval:
                    # nonlinear regression # usually a bit compute-heavy here
                    gskrreg = GridSearchCV(
                        KernelRidge(kernel="rbf", gamma=0.1),
                        param_grid={
                            "alpha": [1e0, 0.1, 1e-2, 1e-3],
                            "gamma": np.logspace(-2, 2, 4),
                        },
                        cv=3,
                        n_jobs=-1,
                    )
                    nonlinear = evaluate_prediction(gskrreg, r2_score, *data)
                # NOTE: MLP is a lightweight alternative
                nonlinear = evaluate_prediction(MLPRegressor(max_iter=1000), r2_score, *data)

                # for discrete factors, do classification and compute accuracy
            else:  # factor_type == "discrete":
                # we disable prediction on zpos in m3di because it is constant
                # logistic classification
                logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
                linear = evaluate_prediction(logreg, accuracy_score, *data)
                # nonlinear classification
                mlpreg = MLPClassifier(max_iter=1000)
                nonlinear = evaluate_prediction(mlpreg, accuracy_score, *data)
            linear_res.append(linear)
            nonlinear_res.append(nonlinear)
        id_score_linear.append(np.stack(linear_res))
        id_score_nonlinear.append(np.stack(nonlinear_res))
    return np.stack(id_score_linear), np.stack(id_score_nonlinear)

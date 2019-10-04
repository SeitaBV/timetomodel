from typing import Dict, Tuple, Optional, Sequence
from datetime import datetime

import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

from timetomodel.speccing import ModelSpecs
from timetomodel.featuring import construct_features
from timetomodel.exceptions import MissingData, UnsupportedModel
from timetomodel import MODEL_TYPES, ModelState


"""
Functions for working with time series models.
"""


def create_fitted_model(
    specs: ModelSpecs,
    version: str,  # TODO: throw out?
    regression_frame: pd.DataFrame = None,
) -> MODEL_TYPES:
    """
    Create a new fitted model with the given specs.
    """
    if regression_frame is None:
        regression_frame = construct_features(time_range="train", specs=specs)

    # Remove any observation where data is missing.
    # Other parts of the workflow cannot handle missing data, so everything should be verified here.
    regression_frame = regression_frame.dropna(axis=1)
    if regression_frame.empty:
        raise MissingData(
            "Missing data (probably one of the regressors contains no data)"
        )

    x_train = regression_frame.drop(columns=[specs.outcome_var.name])
    y_train = regression_frame[specs.outcome_var.name]

    package_str = specs.model_type.__module__.split(".")[0]
    if package_str == "statsmodels":
        model = specs.model_type(y_train, x_train, **specs.model_params)
        fitted_model = model.fit()
        fitted_model.get_params = fitted_model.params
    elif package_str == "sklearn":
        model = specs.model_type(**specs.model_params)
        fitted_model = model.fit(X=x_train, y=y_train)
        fitted_model.params = fitted_model.get_params
    else:
        raise UnsupportedModel("Unknown model type: %s " % specs.model_type)

    return fitted_model


def evaluate_models(
    m1: ModelState, m2: Optional[ModelState] = None, plot_path: str = None
):
    """
    Run a model or two against test data and plot results.
    Useful to judge model performance or compare two models.
    Shows RMSE values, plots error distributions and prints the time it took to forecast.

    TODO: support testing m2 next to m1
    """
    fitted_m1, m1_specs = m1.split()

    regression_frame = construct_features(time_range="test", specs=m1_specs)

    x_test = regression_frame.iloc[:, 1:]
    y_test = np.array(regression_frame.iloc[:, 0])

    try:
        y_hat_test = fitted_m1.predict(x_test)
    except TypeError:
        y_hat_test = fitted_m1.predict(
            start=x_test.index[0], end=x_test.index[-1], exog=x_test
        )

    # Back-transform if the data was transformed
    if m1_specs.outcome_var.feature_transformation is not None:
        y_test = m1_specs.outcome_var.feature_transformation.back_transform_value(
            y_test
        )
        y_hat_test = m1_specs.outcome_var.feature_transformation.back_transform_value(
            y_hat_test
        )

    print(
        "rmse = %s"
        % (str(round(sm.tools.eval_measures.rmse(y_test, y_hat_test, axis=0), 4)))
    )

    plot_true_versus_predicted(
        regression_frame.index, y_test, y_hat_test, None, None, plot_path
    )

    plot_error_graph(y_test, y_hat_test, plot_path=plot_path)


def plot_true_versus_predicted(
    indices: pd.DatetimeIndex,
    true_values: Sequence[float],
    predicted_values: Sequence[float],
    predicted_values_low: Sequence[float] = None,
    predicted_values_high: Sequence[float] = None,
    plot_path: str = None,
):
    """
    Helper function to plot observations, forecasts and prediction intervals.
    """
    plt.plot(indices.to_pydatetime(), true_values, label="actual")
    plt.plot(indices.to_pydatetime(), predicted_values, label="predicted")
    if predicted_values_low is not None and predicted_values_high is not None:
        plt.fill_between(
            indices.to_pydatetime(),
            predicted_values_high,
            predicted_values_low,
            facecolor="orange",
            alpha=0.1,
            label="Confidence interval",
        )
    plt.xlabel("tested time steps")
    plt.ylabel("values")
    plt.legend()
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(plot_path + "/true_versus_predicted.png")
        plt.close()


def plot_error_graph(
    true_values: Sequence[float],
    predicted_values: Sequence[float],
    use_abs_errors: bool = False,
    plot_path: str = None,
):

    results_df = pd.DataFrame({"y_hat_test": predicted_values, "y_test": true_values})

    # remove 0 s
    results_df = results_df[(results_df != 0).all(1)]

    results_df["max_error"] = abs(results_df.y_hat_test / results_df.y_test - 1)
    if use_abs_errors is True:
        #  if you want to look at abs values, instead of (abs)proportional errors
        results_df["max_error"] = abs(results_df.y_hat_test - results_df.y_test)

    results_df.sort_values("max_error", inplace=True)
    results_df["proportion"] = (np.arange(len(results_df)) + 1) / len(results_df)

    plt.plot(results_df["max_error"], results_df["proportion"], "-o")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("max error for proportion")
    plt.ylabel("proportion of observations")
    if plot_path is None:
        plt.show()
    else:
        plt.savefig(plot_path + "/error_graph.png")
        plt.close()


def model_param_grid_search(
    df: pd.DataFrame,
    start_of_training_data: datetime,
    end_of_test_data: datetime,
    params: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:

    """
    Creates and tests models with different model parameters.
    Returns the best parameter set w.r.t. smallest RMSE.
    """
    return {}

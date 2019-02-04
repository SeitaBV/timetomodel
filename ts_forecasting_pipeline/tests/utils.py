from datetime import datetime, timedelta

import pandas as pd
from statsmodels.api import OLS

from ts_forecasting_pipeline import speccing, modelling


def create_dummy_model(data_start: datetime, data_range_in_hours: int) -> modelling.ModelState:
    """
    Create a dummy model. data increases linearly, regressor is constant (useless).
    Use two different ways to define Series specs to test them.
    """
    dt_range = pd.date_range(
        data_start, data_start + timedelta(hours=data_range_in_hours), freq="1H"
    )
    outcome_values = [0]
    regressor_values = [5]
    for i in range(1, len(dt_range)):
        outcome_values.append(outcome_values[i - 1] + 1)
        regressor_values.append(5)
    outcome_series = pd.Series(index=dt_range, data=outcome_values)
    regressor_series = pd.Series(index=dt_range, data=regressor_values)
    specs = modelling.ModelSpecs(
        outcome_var=speccing.ObjectSeriesSpecs(outcome_series, "my_outcome"),
        model=OLS,
        lags=[1, 2],
        frequency=timedelta(hours=1),
        horizon=timedelta(minutes=120),
        remodel_frequency=timedelta(hours=48),
        regressors=[regressor_series],
        start_of_training=data_start + timedelta(hours=2),  # leaving room for NaN in lags
        end_of_testing=data_start + timedelta(hours=int(data_range_in_hours / 3)),
    )
    return modelling.ModelState(
        modelling.create_fitted_model(specs, version="0.1", save=False), specs
    )


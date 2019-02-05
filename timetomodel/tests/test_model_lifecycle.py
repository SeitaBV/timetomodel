from datetime import datetime, timedelta
import os
import pytest
import json

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.api import OLS

from timetomodel import ModelState, speccing, modelling, serializing
from timetomodel.utils.time_utils import get_most_recent_quarter, tz_aware_utc_now

"""
These tests should test model lifecycle, but are not very relevant right now, as that functionality still needs to 
be built (if anyone should actually need it - a model spec can also nicely live as code that can be referred to later
when needed)."""

serializing.MODEL_DIR = "test_models"


@pytest.fixture(scope="module", autouse=True)
def clean_model_dir():
    # before: make sure dir exists and is empty
    if not os.path.exists(serializing.MODEL_DIR):
        os.mkdir(serializing.MODEL_DIR)
    for f in os.listdir(serializing.MODEL_DIR):
        os.remove("%s/%s" % (serializing.MODEL_DIR, f))
    yield
    # after: clean out that dir
    for f in os.listdir(serializing.MODEL_DIR):
        os.remove("%s/%s" % (serializing.MODEL_DIR, f))
    os.rmdir(serializing.MODEL_DIR)


def create_dummy_model(now: datetime, save: bool = False) -> modelling.ModelState:
    """
    Create a dummy model. Try out two different ways to define Series specs.
    """
    now15 = now + timedelta(minutes=15)
    dt_index = pd.date_range(start=now, end=now15, freq="15T")
    specs = modelling.ModelSpecs(
        outcome_var=speccing.ObjectSeriesSpecs(pd.Series(index=dt_index, data={now: 2, now15: 4}), "solar"),
        model=OLS,
        lags=[],
        frequency=timedelta(minutes=15),
        horizon=timedelta(hours=48),
        regressors=[pd.Series(index=dt_index, data={now: 1, now15: 1})],
        start_of_training=now,
        end_of_testing=now15,
    )
    return modelling.ModelState(
        modelling.create_fitted_model(specs, version="0.1", save=save), specs
    )


def test_model_state_params():
    model_state = create_dummy_model(now=get_most_recent_quarter())
    model, specs = model_state.split()
    with pytest.raises(Exception) as e_info:
        ModelState(specs, model)  # wrong order
    assert "model parameter needs to be of type" in str(e_info.value)


@pytest.mark.skip(reason="Currently loading the model type is not working.")
def test_create_and_load_model():
    """Create a model and check if it exists on file. Reload it.
    """
    now = get_most_recent_quarter()
    now15 = now + timedelta(minutes=15)
    create_dummy_model(now=now, save=True)

    assert os.path.exists("%s/solar_0.1.pickle" % serializing.MODEL_DIR)
    assert os.path.exists("%s/specs.json" % serializing.MODEL_DIR)

    # check specs in file
    with open("%s/specs.json" % serializing.MODEL_DIR, "r") as specs_file:
        all_specs = json.loads(specs_file.read())
        assert all_specs["solar_0.1"]["model_type"] == "statsmodels.regression.linear_model.OLS"
        assert (
            json.loads(all_specs["solar_0.1"]["regressors"][0])["name"] == "Regressor1"
        )

    # check via loading function
    model_state = serializing.load_model("solar")
    assert isinstance(model_state.model, RegressionResultsWrapper)
    specs = model_state.specs
    assert specs.creation_time > tz_aware_utc_now() - timedelta(minutes=5)
    assert isinstance(specs.outcome_var, speccing.ObjectSeriesSpecs)
    assert specs.outcome_var.load_series(expected_frequency=timedelta(minutes=15)).index[0] == now
    assert specs.outcome_var.load_series(expected_frequency=timedelta(minutes=15)).index[1] == now15
    assert specs.outcome_var.name == "solar"
    for r in specs.regressors:
        assert isinstance(r, speccing.ObjectSeriesSpecs)
        assert r.load_series(expected_frequency=timedelta(minutes=15)).index[1] == now15
    assert specs.start_of_training < specs.end_of_testing


def test_unsupported_serialisation_of_transformations():
    now = get_most_recent_quarter()
    model_state = create_dummy_model(now=now, save=False)
    model_state.specs.outcome_var.feature_transformation = lambda df: pd.DataFrame()
    with pytest.raises(Exception) as e_info:
        modelling.save_model(model_state, "0.1")
    print(e_info)
    assert "Cannot serialise" in str(e_info.value)


def test_unsupported_serialisation_of_db_series():
    now = get_most_recent_quarter()
    model_state = create_dummy_model(now=now, save=False)
    model_state.specs.regressors = [
        speccing.DBSeriesSpecs(db_engine=None, query=None, name="bla")
    ]
    with pytest.raises(Exception) as e_info:
        modelling.save_model(model_state, "0.1")
    assert "Cannot serialise" in str(e_info.value)

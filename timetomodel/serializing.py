"""Code to help (de)serializing ModelStates"""

import os
import json
import pickle
import copy
from datetime import date, datetime, timedelta, tzinfo
from typing import Type


from statsmodels.base.model import LikelihoodModelResults as StatsModel
import pandas as pd

from timetomodel.speccing import SeriesSpecs, DBSeriesSpecs, ModelSpecs
from timetomodel.exceptions import ModelLocationProblem
from timetomodel import ModelState


MODEL_DIR = "models"


def save_model(model_state: ModelState, version: str):
    """
    Save (a deep copy of) model as a pickle and specs as JSON (NOTE: not thread-safe, of course.
    Best use in single-user mode).
    """
    model, specs = model_state.model, model_state.specs
    if any(
        [specs.outcome_var.feature_transformation]
        + [rs.feature_transformation for rs in specs.regressors]
    ):
        raise Exception(
            "Cannot serialise this ModelSpecs object. Transformation functions are not yet supported."
        )
    if isinstance(specs.outcome_var, DBSeriesSpecs) or any(
        isinstance(r, DBSeriesSpecs) for r in specs.regressors
    ):
        raise Exception(
            "Cannot serialise this ModelSpecs object. DBSeriesSpecs are not yet supported."
        )
    ensure_model_specs_file(MODEL_DIR)
    model_identifier = "%s_%s" % (specs.outcome_var.name, version)
    model_filename = "%s.pickle" % model_identifier
    copy.deepcopy(model).save("%s/%s" % (MODEL_DIR, model_filename), remove_data=True)
    specs.model_filename = model_filename
    with open("%s/specs.json" % MODEL_DIR, "r") as specs_file:
        existing_specs = json.loads(specs_file.read())
    existing_specs[model_identifier] = specs.as_dict()
    existing_specs_json = json.dumps(existing_specs, default=json_serial_helper)
    with open("%s/specs.json" % MODEL_DIR, "w") as specs_file:
        specs_file.write(existing_specs_json)


def load_model(outcome_var_name: str) -> ModelState:
    """
    Load a model from disk (pickle). Use outcome variable in name.
    """
    ensure_model_specs_file(MODEL_DIR)
    specs: ModelSpecs = None
    with open("%s/specs.json" % MODEL_DIR, "r") as specs_file:
        existing_specs = json.loads(specs_file.read())
        for identifier in existing_specs.keys():
            if identifier.startswith(outcome_var_name):
                specs_json = existing_specs[identifier]
                specs = ModelSpecs(**specs_json)
                break
    if specs is None:
        raise ModelLocationProblem(
            "No model found which predicts %s ..." % outcome_var_name
        )
    model = StatsModel.load("%s/%s" % (MODEL_DIR, specs.model_filename))
    return ModelState(model, specs)


def ensure_model_specs_file(model_dir: str):
    """Make sure specs.json is in the model dir"""
    if not os.path.exists(model_dir):
        raise ModelLocationProblem("Cannot find MODEL_DIR: %s" % model_dir)
    filename = "%s/specs.json" % model_dir
    if not os.path.exists(filename):
        with open(filename, "w") as specs_file:
            specs_file.write("{}")


def json_serial_helper(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Type):
        return "%s.%s" % (obj.__module__, obj.__name__)
    if isinstance(obj, tzinfo):
        return str(obj)
    if isinstance(obj, SeriesSpecs):
        return json.dumps(obj.as_dict(), default=json_serial_helper)
    if isinstance(obj, pd.Series):
        return obj.to_json(date_format="iso")
    if callable(obj):
        return pickle.dumps(obj).decode(
            encoding="latin1"
        )  # attention: this dumps only the importable location!
    raise TypeError("Type %s not serializable" % type(obj))

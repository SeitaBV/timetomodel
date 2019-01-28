# Time series forecasting pipeline

[![Build Status](https://travis-ci.com/SeitaBV/ts-forecasting-pipeline.svg?branch=master)](https://travis-ci.com/SeitaBV/ts-forecasting-pipeline)

This repository contains code to support time series forecasting.

It contains functionality that is both useful for data & model exploration
(model training & testing) as well as production code (the code to fit models
and generate forecasts can be re-used)
while the data is now loaded from the DB instead from csv files.

The pipeline has support for:

* Data sources for target and regressor variables (load from Pandas objects, Pandas pickles or databases via SQLAlchemy)
* Lagging
* Timezone awareness support
* Custom data transformations
* Automatic model re-training if validity period is over
* Creating rolling forecasts


## Example

TODO


## Getting Started

### Dependencies using Anaconda

* Install Anaconda for Python3.6+
* Make a virtual environment: `conda create --name tsfp-venv python=3.6`
* Activate it: `source activate tsfp-venv` (or `activate tsfp-venv` for Windows)
* Install dependencies by running setup: `python setup.py develop`

### Notebooks

* If you edit notebooks, make sure results do not end up in git:

      conda install -c conda-forge nbstripout
      nbstripout --install


### Using ts-forecasting-pipeline on a server

On a server, you are not interested in the plotting capabilities, so 


## Glossary

Here is a short list of terms we have been using at Seita so far and would like to keep using in this way:

Term                | Meaning
---                 | ---
Training            | A time interval with known outcomes, regressors and lag variables, which is used to fit the forecast model.
Testing             | A time interval with known outcomes, regressors and lag variables, which is used to test the accuracy of the fitted model. Prevents overfitting.
Training/test ratio | Usually, a single time interval is selected for training and testing, which is split into two parts. The training/test ratio determines where to make this split (the first part is used for training, the last part is used for testing).
Start of training   | Index of the first timeslot on which the model is trained.
End of testing      | Index of the last timeslot on which the model is tested.
Outcome             | The dependent variable, often referred to as `y` (or `yhat` when the outcome is a forecast).
Lag                 | A duration `L` indicating dependency of the outcome `y(t)` on an earlier outcome `y(t-L)` (lags indicate an autoregressive model)
Regressor           | An independent external variable, often referred to as `x`.
Horizon				| The duration between the timeslot at which a forecast is made and the timeslot which is forecast. The horizons are usually fixed for an application. 
Timeslot			| A time interval within a time series with a certain resolution (e.g. 15 minutes, see below). We index timeslots by their start time.
Resolution			| The distance between timeslots used in the application (called "frequency" in pandas dataframes).
Time window			| All timeslots between some start (including) and end time (excluding).
Rolling forecast	| Shorthand for rolling-horizon forecast. Its time window refers to the forecasted time (i.e. the time at which you made the forecast plus the horizon). 
Forecasts			| Shorthand for fixed-horizon forecast.  
Forecast			| Forecast for a single timeslot made during another timeslot.

## Automated tests

High-level tests for forecasting models are stated in *test_models.py*.
Add any new models you want to include in the tests there.
To run the tests, use:

        python -m pytest tests/

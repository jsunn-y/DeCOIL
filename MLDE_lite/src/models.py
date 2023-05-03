from typing import Callable, Dict
import xgboost as xgb
from sklearn import linear_model

#code modified from https://github.com/google-research/slip/blob/main/models.py
class KerasModelWrapper:
    """Wraps a Keras model to have the sklearn model interface."""

    def __init__(self,
                 model_build_fn: Callable,
                 sequence_length: int,
                 vocab_size: int,
                 fit_kwargs: Dict = dict()):
        """Initialize a KerasModelWrapper.
        Args:
          model_build_fn: A function that when called with arguments
            `model_build_fn(sequence_length, vocab_size)` returns a Keras model.
          sequence_length: The length of input sequences.
          vocab_size: The one-hot dimension size for input sequences.
          fit_kwargs: An optional dictionary of keyword arguments passed to the
            Keras model.fit(**fit_kwargs). See
              https://keras.io/api/models/model_training_apis/ for more details.
        """
        self._model_build_fn = model_build_fn
        self._fit_kwargs = fit_kwargs
        self._sequence_length = sequence_length
        self._vocab_size = vocab_size

    def fit(self, X, y):
        self._model = self._model_build_fn(
            self._sequence_length, self._vocab_size)
        self._model.fit(X, y, **self._fit_kwargs)

    def predict(self, X):
        return self._model.predict(x=X, verbose=0).squeeze(axis=1)


def build_linear_model(model_kwargs):
    default_kwargs = {
        'ridge_alpha': 1.0,
        'ridge_fit_intercept': True,
    }
    kwargs = default_kwargs.copy()
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]

    model = linear_model.Ridge(**model_kwargs)
    return model

def build_boosting_model(model_kwargs):
    default_kwargs = {
                "objective": "reg:tweedie",
                "early_stopping_rounds": 10,
                "nthread": -1
                }
    kwargs = default_kwargs.copy()
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]

    model = xgb.XGBRegressor(**model_kwargs)
    return model

def get_model(model_name,
              model_kwargs: Dict):
    """Returns model, flatten_inputs."""
    if model_name == 'ridge':
        return build_linear_model(model_kwargs)
    elif model_name == 'boosting':
        return build_boosting_model(model_kwargs)
    else:
        raise NotImplementedError
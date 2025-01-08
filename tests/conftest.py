# -*- coding: utf-8 -*-
"""Shared pytest fixtures.

All fixtures are scoped to the session, because creating the ``pyfunc_model``
fixture is expensive. requiring read/write to disk. The ``pyfunc_model``
fixture depends on all other fixtures in this module, so they also have to be
scoped to the session. Setting the scope to 'function' slows down running the
complete test suite by a factor of approx 10x (~2s -> 20s)

Copyright (C) 2022, Auto Trader UK

"""

import os.path
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mlflow.models import infer_signature  # type: ignore
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext  # type: ignore
from mlflow.pyfunc import load_model as pyfunc_load_model  # type: ignore
from mlflow.pyfunc import save_model as pyfunc_save_model


class DeepThought(PythonModel):
    """A simple PythonModel that returns `42` for each input row."""

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> npt.ArrayLike:
        return np.array([42] * len(model_input))


class DeepThoughtSeries(PythonModel):
    """A PythonModel that returns a DataFrame."""

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> pd.Series:
        return pd.Series([42] * len(model_input))


class DeepThoughtDataframe(PythonModel):
    """A PythonModel that returns a DataFrame."""

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "answer": 42,
                    "question": (
                        "The ultimate question of life, the universe, "
                        "and everything!"
                    ),
                }
            ]
            * len(model_input)
        )


class NaNModel(PythonModel):
    """A PythonModel that returns NaN."""

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame):
        return np.array([np.nan] * len(model_input))


class NaNModelSeries(PythonModel):
    """A PythonModel that returns NaNs in a Series."""

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> pd.Series:
        return pd.Series(NaNModel().predict(context, model_input))


class NaNModelDataFrame(PythonModel):
    """A PythonModel that returns NaNs in a DataFrame."""

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> pd.DataFrame:
        nan_model = NaNModelSeries()
        return pd.DataFrame(
            {
                "a": nan_model.predict(context, model_input),
                "b": nan_model.predict(context, model_input),
            }
        )


class StrModel(PythonModel):
    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> npt.ArrayLike:
        return np.array(["42"] * len(model_input))


class StrModelSeries(PythonModel):
    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> pd.Series:
        return pd.Series(StrModel().predict(context, model_input))


class ExceptionRaiser(PythonModel):
    """A PythonModle that always raises an exception."""

    ERROR_MESSAGE = "I always raise an error!"

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> pd.DataFrame:
        raise ValueError(self.ERROR_MESSAGE)


@pytest.fixture(scope="session")
def model_input() -> pd.DataFrame:
    int32_ = [np.int32(i) for i in range(5)]
    int64_ = [np.int64(i) for i in range(5)]
    double_ = [np.double(i) for i in range(5)]
    bool_ = [np.bool_(i) for i in range(5)]
    bytes_ = [bytes(i) for i in range(5)]
    str_ = [str(i) for i in range(5)]
    datetime_ = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(5)]
    return pd.DataFrame(
        dict(
            int32=int32_,
            int64=int64_,
            double=double_,
            bool=bool_,
            bytes=bytes_,
            str=str_,
            datetime=datetime_,
        )
    )


def _get_pyfunc_model(
    python_model: PythonModel,
    model_input: pd.DataFrame,
    model_output: Any,
) -> PyFuncModel:
    signature = infer_signature(
        model_input=model_input,
        model_output=model_output,
    )
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, python_model.__class__.__name__)
        pyfunc_save_model(
            model_path,
            python_model=python_model,
            signature=signature,
        )
        yield pyfunc_load_model(model_path)


# Model with ndarray output


@pytest.fixture(scope="session")
def python_model_ndarray() -> PythonModel:
    return DeepThought()


@pytest.fixture(scope="session")
def model_output_ndarray(
    python_model_ndarray: PythonModel, model_input: pd.DataFrame
) -> npt.ArrayLike:
    return python_model_ndarray.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_ndarray(
    python_model_ndarray: PythonModel,
    model_input: pd.DataFrame,
    model_output_ndarray: npt.ArrayLike,
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_ndarray, model_input, model_output_ndarray
    )


# Model with Series output


@pytest.fixture(scope="session")
def python_model_series() -> PythonModel:
    """Model with Series output."""
    return DeepThoughtSeries()


@pytest.fixture(scope="session")
def model_output_series(
    python_model_series: PythonModel, model_input: pd.DataFrame
) -> pd.Series:
    return python_model_series.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_series(
    python_model_series: PythonModel,
    model_input: pd.DataFrame,
    model_output_series: pd.Series,
) -> PyFuncModel:
    yield from _get_pyfunc_model(python_model_series, model_input, model_output_series)


# Model with DataFrame output


@pytest.fixture(scope="session")
def python_model_dataframe() -> PythonModel:
    """Model with DataFrame output."""
    return DeepThoughtDataframe()


@pytest.fixture(scope="session")
def model_output_dataframe(
    python_model_dataframe: PythonModel, model_input: pd.DataFrame
) -> npt.ArrayLike:
    return python_model_dataframe.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_dataframe(
    python_model_dataframe: PythonModel,
    model_input: pd.DataFrame,
    model_output_dataframe: pd.DataFrame,
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_dataframe, model_input, model_output_dataframe
    )


# Models that can output NaNs


@pytest.fixture(scope="session")
def python_model_nan_ndarray() -> PythonModel:
    """Model with ndarray output of NaNs."""
    return NaNModel()


@pytest.fixture(scope="session")
def model_output_nan_ndarray(
    python_model_nan_ndarray, model_input: pd.DataFrame
) -> npt.ArrayLike:
    return python_model_nan_ndarray.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_nan_ndarray(
    python_model_nan_ndarray: PythonModel,
    model_input: pd.DataFrame,
    model_output_ndarray: npt.ArrayLike,  # Use to infer correct
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_nan_ndarray, model_input, model_output_ndarray
    )


@pytest.fixture(scope="session")
def python_model_nan_series() -> PythonModel:
    return NaNModelSeries()


@pytest.fixture(scope="session")
def model_output_nan_series(
    python_model_nan_series, model_input: pd.DataFrame
) -> pd.Series:
    return python_model_nan_series.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_nan_series(
    python_model_nan_series: PythonModel,
    model_input: pd.DataFrame,
    model_output_series: pd.Series,  # Use to infer correct
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_nan_series, model_input, model_output_series
    )


@pytest.fixture(scope="session")
def python_model_nan_dataframe() -> PythonModel:
    return NaNModelDataFrame()


@pytest.fixture(scope="session")
def model_output_nan_dataframe(
    python_model_nan_dataframe, model_input: pd.DataFrame
) -> pd.DataFrame:
    return python_model_nan_dataframe.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_nan_dataframe(
    python_model_nan_dataframe: PythonModel,
    model_input: pd.DataFrame,
    model_output_nan_dataframe: pd.DataFrame,  # Use to infer correct
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_nan_dataframe, model_input, model_output_nan_dataframe
    )


# Models that returns a sequence of strings
@pytest.fixture(scope="session")
def python_model_str_ndarray() -> PythonModel:
    return StrModel()


@pytest.fixture(scope="session")
def model_output_str_ndarray(
    python_model_str_ndarray, model_input: pd.DataFrame
) -> npt.ArrayLike:
    return python_model_str_ndarray.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_str_ndarray(
    python_model_str_ndarray: PythonModel,
    model_input: pd.DataFrame,
    model_output_str_ndarray: npt.ArrayLike,  # Use to infer correct
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_str_ndarray, model_input, model_output_str_ndarray
    )


@pytest.fixture(scope="session")
def python_model_str_series() -> PythonModel:
    return StrModelSeries()


@pytest.fixture(scope="session")
def model_output_str_series(
    python_model_str_series, model_input: pd.DataFrame
) -> pd.Series:
    return python_model_str_series.predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def pyfunc_model_str_series(
    python_model_str_series: PythonModel,
    model_input: pd.DataFrame,
    model_output_str_series: pd.Series,  # Use to infer correct
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_str_series, model_input, model_output_str_series
    )


# Model that always raises exceptions
@pytest.fixture(scope="session")
def python_model_value_error() -> PythonModel:
    return ExceptionRaiser()


@pytest.fixture(scope="session")
def pyfunc_model_value_error(
    python_model_value_error,
    model_input: pd.DataFrame,
    model_output_series: pd.Series,  # Use to infer correct
) -> PyFuncModel:
    yield from _get_pyfunc_model(
        python_model_value_error, model_input, model_output_series
    )


@pytest.fixture(
    params=[
        "ndarray",
        "series",
        "dataframe",
        "nan_ndarray",
        "nan_series",
        "nan_dataframe",
        "str_ndarray",
        "str_series",
    ]
)
def pyfunc_model(request: pytest.FixtureRequest) -> PyFuncModel:
    return request.getfixturevalue(f"pyfunc_model_{request.param}")  # type: ignore
    # param is an optional attribute, and may not be present when type checking
    # https://docs.pytest.org/en/6.2.x/reference.html#pytest.FixtureRequest

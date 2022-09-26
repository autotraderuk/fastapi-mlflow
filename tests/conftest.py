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

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mlflow.models import infer_signature  # type: ignore
from mlflow.pyfunc import (  # type: ignore
    PyFuncModel,
    PythonModel,
    PythonModelContext,
    load_model as pyfunc_load_model,
    save_model as pyfunc_save_model,
)


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


@pytest.fixture(scope="session")
def model_input() -> pd.DataFrame:
    a = [np.int64(i) for i in range(5)]
    b = [np.double(i) for i in range(5)]
    c = [np.bool_(i) for i in range(5)]
    d = [bytes(i) for i in range(5)]
    e = [str(i) for i in range(5)]
    f = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(5)]
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e, f=f))


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
    signature = infer_signature(
        model_input=model_input,
        model_output=model_output_ndarray,
    )
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model_ndarray")
        pyfunc_save_model(
            model_path,
            python_model=python_model_ndarray,
            signature=signature,
        )
        yield pyfunc_load_model(model_path)


# Model with Series output


@pytest.fixture(scope="session")
def python_model_series() -> PythonModel:
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
    signature = infer_signature(
        model_input=model_input,
        model_output=model_output_series,
    )
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model_series")
        pyfunc_save_model(
            model_path,
            python_model=python_model_series,
            signature=signature,
        )
        yield pyfunc_load_model(model_path)


# Model with DataFrame output


@pytest.fixture(scope="session")
def python_model_dataframe() -> PythonModel:
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
    signature = infer_signature(
        model_input=model_input,
        model_output=model_output_dataframe,
    )
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model_dataframe")
        pyfunc_save_model(
            model_path,
            python_model=python_model_dataframe,
            signature=signature,
        )
        yield pyfunc_load_model(model_path)


@pytest.fixture(params=["ndarray", "series", "dataframe"])
def pyfunc_model(request: pytest.FixtureRequest) -> PyFuncModel:
    return request.getfixturevalue(f"pyfunc_model_{request.param}")  # type: ignore
    # param is an optional attribute, and may not be present when type checking
    # https://docs.pytest.org/en/6.2.x/reference.html#pytest.FixtureRequest

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

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.models import ModelSignature, infer_signature
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext


class DeepThought(PythonModel):
    """A simple PythonModel that always returns `42`."""

    def predict(
        self, context: PythonModelContext, model_input: pd.DataFrame
    ) -> np.array:
        return np.array([42] * len(model_input))


@pytest.fixture(scope="session")
def model_input() -> pd.DataFrame:
    a = [np.int64(i) for i in range(5)]
    b = [np.double(i) for i in range(5)]
    c = [np.bool_(i) for i in range(5)]
    d = [bytes(i) for i in range(5)]
    e = [str(i) for i in range(5)]
    f = [datetime(2022, 1, 1) + timedelta(days=i) for i in range(5)]
    return pd.DataFrame(dict(a=a, b=b, c=c, d=d, e=e, f=f))


@pytest.fixture(scope="session")
def model_output(model_input: pd.DataFrame) -> np.array:
    return DeepThought().predict(context=None, model_input=model_input)


@pytest.fixture(scope="session")
def model_signature(
    model_input: pd.DataFrame, model_output: np.array
) -> ModelSignature:
    return infer_signature(
        model_input=model_input,
        model_output=model_output,
    )


@pytest.fixture(scope="session")
def pyfunc_model(model_signature: ModelSignature) -> PyFuncModel:
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model")
        mlflow.pyfunc.save_model(
            model_path, python_model=DeepThought(), signature=model_signature
        )
        yield mlflow.pyfunc.load_model(model_path)

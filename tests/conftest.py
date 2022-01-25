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
    return pd.DataFrame(dict(a=list(range(5)), b=list(range(5))))


@pytest.fixture(scope="session")
def model_output() -> np.array:
    return np.array([42] * 5)


@pytest.fixture(scope="session")
def signature(model_input: pd.DataFrame, model_output: np.array) -> ModelSignature:
    return infer_signature(
        model_input=model_input,
        model_output=model_output,
    )


@pytest.fixture(scope="session")
def pyfunc_model(signature: ModelSignature) -> PyFuncModel:
    with TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model")
        mlflow.pyfunc.save_model(
            model_path, python_model=DeepThought(), signature=signature
        )
        yield mlflow.pyfunc.load_model(model_path)

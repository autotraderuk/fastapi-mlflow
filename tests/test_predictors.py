# -*- coding: utf-8 -*-
"""Document test_predictors here.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import isawaitable, signature
from unittest.mock import Mock

import pandas as pd
import pydantic
import pytest
from mlflow.pyfunc import PyFuncModel

from fastapi_mlflow.predictors import build_predictor


def test_build_predictor_return_type(pyfunc_model: PyFuncModel):
    """build_predictor should return a callable that returns an awaitable."""
    predictor = build_predictor(pyfunc_model)

    assert callable(predictor)
    assert isawaitable(predictor(Mock(spec_set=pydantic.BaseModel)))


def test_build_predictor_return_has_correct_signature_for_input(
    pyfunc_model: PyFuncModel, model_input: pd.DataFrame
):
    sig = signature(build_predictor(pyfunc_model))

    assert "request" in sig.parameters
    request_type = sig.parameters["request"].annotation
    assert issubclass(
        request_type, pydantic.BaseModel
    ), "type of `request` parameter of predictor function is not a subclass of pydantic.BaseModel"
    assert isinstance(
        request_type(**model_input.to_dict(orient="records")[0]), request_type
    ), "type derived from predictor function signature cannot be instantiated with expected input data"
    with pytest.raises(pydantic.ValidationError):
        request_type(foo="bar")


def test_build_predictor_return_has_correct_signature_for_return_type(
    pyfunc_model: PyFuncModel,
):
    sig = signature(build_predictor(pyfunc_model))
    return_type = sig.return_annotation
    assert issubclass(
        return_type, pydantic.BaseModel
    ), "return type annotation of predictor function is not a subclass of pydantic.BaseModel"

# -*- coding: utf-8 -*-
"""Document test_predictors here.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import isawaitable, signature
from typing import get_args, get_origin

import pandas as pd
import pydantic
import pytest
from mlflow.pyfunc import PyFuncModel

from fastapi_mlflow.predictors import build_predictor


def test_build_predictor_returns_coroutine_function(
    pyfunc_model: PyFuncModel,
):
    predictor = build_predictor(pyfunc_model)

    assert callable(predictor)
    dummy_request = None
    assert isawaitable(predictor(dummy_request))


def test_coroutine_function_has_correct_signature_for_input(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    assert "request" in sig.parameters
    request_type = sig.parameters["request"].annotation
    assert get_origin(request_type) is list
    assert issubclass(get_args(request_type)[0], pydantic.BaseModel), (
        "type for items in predictor coroutine function parameter `request` is not a subclass of "
        "pydantic.BaseModel"
    )


def test_coroutine_function_signature_items_can_be_constructed(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    request_type = sig.parameters["request"].annotation
    pydantic_request_model = get_args(request_type)[0]
    assert isinstance(
        pydantic_request_model(**model_input.to_dict(orient="records")[0]),
        pydantic_request_model,
    ), (
        "type for items in predictor coroutine function parameter `request` cannot be "
        "constructed with expected input data"
    )
    with pytest.raises(pydantic.ValidationError):
        pydantic_request_model(foo="bar")


def test_coroutine_function_signature_items_raise_validation_error_given_invalid_arguments(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    request_type = sig.parameters["request"].annotation
    pydantic_request_model = get_args(request_type)[0]
    with pytest.raises(pydantic.ValidationError):
        pydantic_request_model(foo="bar")


def test_coroutine_function_has_correct_return_type(
    pyfunc_model: PyFuncModel,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    return_type = sig.return_annotation
    assert get_origin(return_type) is list
    pydantic_return_model = get_args(return_type)[0]
    assert issubclass(
        pydantic_return_model, pydantic.BaseModel
    ), "type of items in return for predictor coroutine function is not a subclass of pydantic.BaseModel"

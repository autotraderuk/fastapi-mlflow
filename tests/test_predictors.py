# -*- coding: utf-8 -*-
"""Test predictor function building.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import signature
from typing import get_args, get_origin

import numpy as np
import pandas as pd
import pydantic
import pytest
from mlflow.pyfunc import PyFuncModel

from fastapi_mlflow.predictors import build_predictor


def test_build_predictor_returns_callable(
    pyfunc_model: PyFuncModel,
):
    predictor = build_predictor(pyfunc_model)

    assert callable(predictor)


def test_predictor_has_correct_signature_for_input(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    assert "request" in sig.parameters
    request_type = sig.parameters["request"].annotation
    assert get_origin(request_type) is list
    assert issubclass(get_args(request_type)[0], pydantic.BaseModel), (
        "type for items in predictor function parameter `request` is not a"
        "subclass of pydantic.BaseModel"
    )


def test_predictor_signature_items_can_be_constructed(
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
        "type for items in predictor function parameter `request` cannot be "
        "constructed with expected input data"
    )
    with pytest.raises(pydantic.ValidationError):
        pydantic_request_model(foo="bar")


def test_predictor_signature_items_raise_validation_error_given_invalid_arguments(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    request_type = sig.parameters["request"].annotation
    pydantic_request_model = get_args(request_type)[0]
    with pytest.raises(pydantic.ValidationError):
        pydantic_request_model(foo="bar")


def test_predictor_has_correct_return_type(
    pyfunc_model: PyFuncModel,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    return_type = sig.return_annotation
    assert get_origin(return_type) is list
    pydantic_return_model = get_args(return_type)[0]
    assert issubclass(pydantic_return_model, pydantic.BaseModel), (
        "type of items in return for predictor function is not a subclass of"
        "pydantic.BaseModel"
    )


def test_predictor_correctly_applies_model(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
    model_output: np.array,
):
    predictor = build_predictor(pyfunc_model)

    request_type = signature(predictor).parameters["request"].annotation
    pydantic_request_model = get_args(request_type)[0]
    request = [
        pydantic_request_model(**row) for row in model_input.to_dict(orient="record")
    ]
    response = predictor(request)
    predictions = [item.prediction for item in response]
    assert predictions == model_output.tolist()

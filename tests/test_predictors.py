# -*- coding: utf-8 -*-
"""Test predictor function building.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import signature
from typing import Union

import numpy.typing as npt
import pandas as pd
import pydantic
import pytest
from mlflow.pyfunc import PyFuncModel  # type: ignore

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
    assert issubclass(request_type, pydantic.BaseModel), (
        "type in predictor function parameter `request` is not a"
        "subclass of pydantic.BaseModel"
    )
    schema = request_type.schema()
    assert "data" in schema["required"]
    assert schema["properties"]["data"]["type"] == "array"
    assert "RequestRow" in schema["definitions"]
    assert schema["definitions"]["RequestRow"]["required"] == list(model_input.columns)


def test_predictor_signature_type_can_be_constructed(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    request_type = sig.parameters["request"].annotation

    instance = request_type(data=model_input.to_dict(orient="records"))

    assert isinstance(instance, request_type), (
        "type predictor function parameter `request` cannot be "
        "constructed with expected input data"
    )
    with pytest.raises(pydantic.ValidationError):
        request_type(foo="bar")


def test_predictor_signature_items_raise_validation_error_given_invalid_arguments(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    request_type = sig.parameters["request"].annotation
    with pytest.raises(pydantic.ValidationError):
        request_type(data=[{"foo": "bar"}])


def test_predictor_has_correct_return_type(
    pyfunc_model: PyFuncModel,
):
    predictor = build_predictor(pyfunc_model)

    sig = signature(predictor)
    return_type = sig.return_annotation
    assert issubclass(return_type, pydantic.BaseModel), (
        "type in predictor function parameter `request` is not a"
        "subclass of pydantic.BaseModel"
    )
    schema = return_type.schema()
    assert "data" in schema["required"]
    assert schema["properties"]["data"]["type"] == "array"
    assert "ResponseRow" in schema["definitions"]


@pytest.mark.parametrize(
    "pyfunc_output_type",
    ["ndarray", "series", "dataframe"],
)
def test_predictor_correctly_applies_model(
    model_input: pd.DataFrame,
    pyfunc_output_type: str,
    request: pytest.FixtureRequest,
):
    pyfunc_model: PyFuncModel = request.getfixturevalue(
        f"pyfunc_model_{pyfunc_output_type}"
    )
    model_output: Union[npt.ArrayLike, pd.DataFrame] = request.getfixturevalue(
        f"model_output_{pyfunc_output_type}"
    )

    predictor = build_predictor(pyfunc_model)

    request_type = signature(predictor).parameters["request"].annotation
    request_obj = request_type(data=model_input.to_dict(orient="records"))
    response = predictor(request_obj)
    try:
        assert response.data == model_output.to_dict(orient="records")  # type: ignore
    except (AttributeError, TypeError):
        predictions = [item.prediction for item in response.data]
        assert predictions == model_output.tolist()  # type: ignore


@pytest.mark.parametrize(
    "pyfunc_output_type",
    ["ndarray", "series", "dataframe"],
)
def test_predictor_handles_model_returning_nan(
    model_input: pd.DataFrame,
    pyfunc_output_type: str,
    request: pytest.FixtureRequest,
):
    pyfunc_model: PyFuncModel = request.getfixturevalue(
        f"pyfunc_model_nan_{pyfunc_output_type}"
    )

    predictor = build_predictor(pyfunc_model)

    request_type = signature(predictor).parameters["request"].annotation
    request_obj = request_type(data=model_input.to_dict(orient="records"))
    response = predictor(request_obj)
    for item in response.data:
        if pyfunc_output_type in ("ndarray", "series"):
            assert item.prediction is None
        else:
            assert item.a is None
            assert item.b is None

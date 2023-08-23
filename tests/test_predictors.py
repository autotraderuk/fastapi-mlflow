# -*- coding: utf-8 -*-
"""Test predictor function building.

Copyright (C) 2022, Auto Trader UK

"""
from datetime import datetime
from inspect import signature
from typing import Union
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pydantic
import pytest
from mlflow.pyfunc import PyFuncModel  # type: ignore

from fastapi_mlflow.exceptions import DictSerialisableException
from fastapi_mlflow.predictors import build_predictor, convert_predictions_to_python


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
    schema = request_type.model_json_schema()
    assert "data" in schema["required"]
    assert schema["properties"]["data"]["type"] == "array"
    assert "RequestRow" in schema["$defs"]
    assert schema["$defs"]["RequestRow"]["required"] == list(model_input.columns)


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
    schema = return_type.model_json_schema()
    assert "data" in schema["required"]
    assert schema["properties"]["data"]["type"] == "array"
    assert "ResponseRow" in schema["$defs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pyfunc_output_type",
    ["ndarray", "series", "dataframe"],
)
async def test_predictor_correctly_applies_model(
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
    response = await predictor(request_obj)
    try:
        assert [row.model_dump() for row in response.data] == model_output.to_dict(orient="records")  # type: ignore
    except (AttributeError, TypeError):
        predictions = [item.prediction for item in response.data]
        assert predictions == model_output.tolist()  # type: ignore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pyfunc_output_type",
    ["ndarray", "series", "dataframe"],
)
async def test_predictor_handles_model_returning_nan(
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
    response = await predictor(request_obj)
    for item in response.data:
        if pyfunc_output_type in ("ndarray", "series"):
            assert item.prediction is None
        else:
            assert item.a is None
            assert item.b is None


@pytest.mark.asyncio
async def test_predictor_raises_custom_wrapped_exception_on_model_error(
    model_input: pd.DataFrame, pyfunc_model_value_error: PyFuncModel
):
    predictor = build_predictor(pyfunc_model_value_error)

    request_type = signature(predictor).parameters["request"].annotation
    request_obj = request_type(data=model_input.to_dict(orient="records"))

    with pytest.raises(DictSerialisableException):
        await predictor(request_obj)


@pytest.mark.asyncio
async def test_predictor_raises_custom_wrapped_exception_on_model_output_conversion(
    model_input: pd.DataFrame,
    pyfunc_model_ndarray: PyFuncModel,
):
    predictor = build_predictor(pyfunc_model_ndarray)

    request_type = signature(predictor).parameters["request"].annotation
    request_obj = request_type(data=model_input.to_dict(orient="records"))

    with patch.object(pyfunc_model_ndarray, "predict") as predict:
        predict.return_value = ["Fail!"]
        with pytest.raises(DictSerialisableException):
            await predictor(request_obj) * len(model_input)


def test_convert_predictions_to_python_ndarray():
    predictions = np.array([1, 2, 3, np.nan])
    response_data = convert_predictions_to_python(predictions)
    assert [
        {"prediction": 1},
        {"prediction": 2},
        {"prediction": 3},
        {"prediction": None},
    ] == response_data


def test_convert_predictions_to_python_ndarray_strings():
    predictions = np.array(["foo", "bar", None])
    response_data = convert_predictions_to_python(predictions)
    assert [
        {"prediction": "foo"},
        {"prediction": "bar"},
        {"prediction": None},
    ] == response_data


def test_convert_predictions_to_python_ndarray_datetimes():
    predictions = np.array([datetime(2023, 1, 1), None])
    response_data = convert_predictions_to_python(predictions)
    assert [
        {"prediction": datetime(2023, 1, 1)},
        {"prediction": None},
    ] == response_data


def test_convert_predictions_to_python_series():
    predictions = pd.Series([1, 2, 3, np.nan])
    response_data = convert_predictions_to_python(predictions)
    assert [
        {"prediction": 1},
        {"prediction": 2},
        {"prediction": 3},
        {"prediction": None},
    ] == response_data


def test_convert_predictions_to_python_dataframe():
    predictions = pd.DataFrame(
        {"expected": [1, 2, 3, np.nan], "confidence": [0.0, 0.5, 1.0, None]}
    )
    response_data = convert_predictions_to_python(predictions)
    assert [
        {"expected": 1, "confidence": 0.0},
        {"expected": 2, "confidence": 0.5},
        {"expected": 3, "confidence": 1.0},
        {"expected": None, "confidence": None},
    ] == response_data

# -*- coding: utf-8 -*-
"""Test application building.

Copyright (C) 2022, Auto Trader UK

"""

from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest as pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.pyfunc import PyFuncModel, PythonModel  # type: ignore

from fastapi_mlflow.applications import build_app


def test_build_app_returns_an_application(pyfunc_model: PyFuncModel):
    app = build_app(pyfunc_model)

    assert isinstance(app, FastAPI), "build_app does not return a FastAPI instance"


def test_build_app_provides_docs(pyfunc_model: PyFuncModel):
    app = build_app(pyfunc_model)

    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200


def test_build_app_has_predictions_endpoint(pyfunc_model: PyFuncModel):
    app = build_app(pyfunc_model)

    client = TestClient(app)
    response = client.post("/predictions", data={})
    assert response.status_code != 404


@pytest.mark.parametrize(
    "pyfunc_output_type",
    ["ndarray", "series", "dataframe"],
)
def test_build_app_returns_good_predictions(
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

    app = build_app(pyfunc_model)
    client = TestClient(app)
    df_str = model_input.to_json(orient="records")
    request_data = f'{{"data": {df_str}}}'

    response = client.post("/predictions", content=request_data)

    assert response.status_code == 200
    results = response.json()["data"]
    try:
        assert model_output.to_dict(orient="records") == results  # type: ignore
    except (AttributeError, TypeError):
        assert [{"prediction": v} for v in np.nditer(model_output)] == results  # type: ignore


def test_built_application_handles_model_exceptions(
    model_input: pd.DataFrame, pyfunc_model_value_error: PyFuncModel
):
    app = build_app(pyfunc_model_value_error)
    client = TestClient(app, raise_server_exceptions=False)
    df_str = model_input.to_json(orient="records")
    request_data = f'{{"data": {df_str}}}'

    response = client.post("/predictions", content=request_data)

    assert response.status_code == 500
    assert {
        "name": "ValueError",
        "message": "I always raise an error!",
    } == response.json()


def test_built_application_logs_exceptions(
    model_input: pd.DataFrame,
    pyfunc_model_value_error: PyFuncModel,
    python_model_value_error: PythonModel,
    caplog: pytest.LogCaptureFixture,
):
    app = build_app(pyfunc_model_value_error)
    client = TestClient(app, raise_server_exceptions=False)
    df_str = model_input.to_json(orient="records")
    request_data = f'{{"data": {df_str}}}'

    _ = client.post("/predictions", content=request_data)

    assert len(caplog.records) >= 1
    log_record = caplog.records[-1]
    assert log_record.name == "fastapi_mlflow.applications"
    assert log_record.message == python_model_value_error.ERROR_MESSAGE


@pytest.mark.parametrize(
    "req_id_header_name",
    [
        "x-request-id",
        "X-Request-Id",
        "X-Request-ID",
        "X-REQUEST-ID",
    ],
)
def test_built_application_logs_exceptions_including_request_id_header_when_sent(
    model_input: pd.DataFrame,
    pyfunc_model_value_error: PyFuncModel,
    python_model_value_error: PythonModel,
    caplog: pytest.LogCaptureFixture,
    req_id_header_name: str,
):
    app = build_app(pyfunc_model_value_error)
    client = TestClient(app, raise_server_exceptions=False)
    df_str = model_input.to_json(orient="records")
    request_data = f'{{"data": {df_str}}}'
    request_id = "abcdef"

    _ = client.post(
        "/predictions", content=request_data, headers={req_id_header_name: request_id}
    )

    log_record = caplog.records[-1]
    assert hasattr(log_record, "x-request-id")
    assert getattr(log_record, "x-request-id") == request_id

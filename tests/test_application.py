# -*- coding: utf-8 -*-
"""Test application building.

Copyright (C) 2022, Auto Trader UK

"""
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.pyfunc import PyFuncModel

from fastapi_mlflow.applications import build_app


def test_build_app_returns_an_application(pyfunc_model: PyFuncModel):
    app = build_app(pyfunc_model)

    assert isinstance(
        app, FastAPI
    ), "build_app does not return a FastAPI instance"


def test_build_app_provides_docs(pyfunc_model: PyFuncModel):
    app = build_app(pyfunc_model)

    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200


def test_build_app_has_predictions_endpoint(pyfunc_model: PyFuncModel):
    app = build_app(pyfunc_model)

    client = TestClient(app)
    response = client.post("/predictions", {})
    assert response.status_code != 404


def test_build_app_returns_good_predictions(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
    model_output: np.array,
):
    app = build_app(pyfunc_model)

    client = TestClient(app)
    request_data = model_input.to_json(orient="records")
    response = client.post("/predictions", data=request_data)
    assert response.status_code == 200
    assert response.json() == [{"prediction": v} for v in model_output]

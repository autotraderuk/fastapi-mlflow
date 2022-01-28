# -*- coding: utf-8 -*-
"""Build FastAPI applications for mlflow model predictions.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import signature

from fastapi import FastAPI
from mlflow.pyfunc import PyFuncModel  # type: ignore

from fastapi_mlflow.predictors import build_predictor


def build_app(pyfunc_model: PyFuncModel) -> FastAPI:
    """Build and return a FastAPI app for the mlflow model."""
    app = FastAPI()
    predictor = build_predictor(pyfunc_model)
    response_model = signature(predictor).return_annotation
    app.add_api_route(
        "/predictions",
        predictor,
        response_model=response_model,
        methods=["POST"],
    )
    return app

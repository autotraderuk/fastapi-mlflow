# -*- coding: utf-8 -*-
"""Build FastAPI applications for mlflow model predictions.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import signature

from fastapi import (
    FastAPI,
    Request,
)
from fastapi.responses import JSONResponse

from mlflow.pyfunc import PyFuncModel  # type: ignore
from fastapi_mlflow.predictors import build_predictor, PyFuncModelPredictError


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

    @app.exception_handler(Exception)
    def handle_exception(_: Request, exc: PyFuncModelPredictError):
        return JSONResponse(
            status_code=500,
            content=exc.to_dict(),
        )

    return app

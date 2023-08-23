# -*- coding: utf-8 -*-
"""Build FastAPI applications for mlflow model predictions.

Copyright (C) 2022, Auto Trader UK

"""
from inspect import signature

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from mlflow.pyfunc import PyFuncModel  # type: ignore

from fastapi_mlflow.exceptions import DictSerialisableException
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
        response_class=ORJSONResponse,
        methods=["POST"],
    )

    @app.exception_handler(DictSerialisableException)
    def handle_serialisable_exception(
        _: Request, exc: DictSerialisableException
    ) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=500,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    def handle_exception(req: Request, exc: Exception) -> ORJSONResponse:
        return handle_serialisable_exception(
            req, DictSerialisableException.from_exception(exc)
        )

    return app

# -*- coding: utf-8 -*-
"""Build functions to wrap mlflow models.

Existing implementation returns a synchronous/blocking function. This decision
was taken because applying ML models is probably CPU-bound. Not having
unnecessary asynchronous code also makes testing simpler.

Current supports only the pyfunc flavour.

Copyright (C) 2022, Auto Trader UK

"""
from typing import Any, Callable, List

import pandas as pd
from mlflow.pyfunc import PyFuncModel  # type: ignore
from pydantic import BaseModel

import fastapi_mlflow._mlflow_types as _mlflow_types


def build_predictor(model: PyFuncModel) -> Callable[[List[BaseModel]], Any]:
    """Build and return a function that wraps the mlflow model.

    Currently supports only the `pyfunc`_ flavour of mlflow.

    :param model: PyFuncModel
    :return: Function suitable for mounting as a FastAPI endpoint or route.

    Example::

        model = load_model("/Users/me/path/to/local/model")
        predictor = build_predictor(model)

    .. _pyfunc: https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html

    """
    # Some of the types here are too dynamic for type checking

    class Request(BaseModel):
        data: List[_mlflow_types.build_input_model(model.metadata.get_input_schema())]  # type: ignore

    class Response(BaseModel):
        data: List[_mlflow_types.build_output_model(model.metadata.get_output_schema())]  # type: ignore

    def predictor(request: Request) -> Response:
        df = pd.DataFrame([row.dict() for row in request.data], dtype=object)
        return Response(data=[{"prediction": row} for row in model.predict(df)])

    return predictor  # type: ignore

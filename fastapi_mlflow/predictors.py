# -*- coding: utf-8 -*-
"""Document predictors here.

Copyright (C) 2022, Auto Trader UK

"""
from typing import Any, Callable, List

import pandas as pd
from mlflow.pyfunc import PyFuncModel
from pydantic import BaseModel

import fastapi_mlflow._mlflow_types as _mlflow_types


def build_predictor(model: PyFuncModel) -> Callable[[BaseModel], Any]:
    """Build and return a coroutine function the wraps the mlflow model.

    Currently supports only the pyfunc flavour.

    :param model: PyFuncModel
    :return: Coroutine function suitable for mounting as a FastAPI endpoint or route.

    Example::

        model = load_model("/Users/me/path/to/local/model")
        predictor = build_predictor(model)

    """
    request_type = _mlflow_types.build_input_model(model.metadata.get_input_schema())
    return_type = _mlflow_types.build_output_model(model.metadata.get_output_schema())

    async def predictor(request: List[request_type]) -> List[return_type]:
        df = pd.DataFrame([row.dict() for row in request], dtype=object)
        return [return_type(prediction=row) for row in model.predict(df)]

    return predictor

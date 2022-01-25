# -*- coding: utf-8 -*-
"""Document predictors here.

Copyright (C) 2022, Auto Trader UK

"""
from typing import Any, Callable

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
    request_type = _mlflow_types.build_input_model(
        model.metadata.get_input_schema()
    )
    return_type = _mlflow_types.build_output_model(
        model.metadata.get_output_schema()
    )

    async def predictor(request: request_type) -> return_type:
        pass

    return predictor

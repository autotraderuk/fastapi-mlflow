# -*- coding: utf-8 -*-
"""Build functions to wrap mlflow models.

Existing implementation returns a synchronous/blocking function. This decision
was taken because applying ML models is probably CPU-bound. Not having
unnecessary asynchronous code also makes testing simpler.

Current supports only the pyfunc flavour.

Copyright (C) 2022, Auto Trader UK

"""
from typing import Any, Callable, List, no_type_check, Dict

import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel  # type: ignore
from pydantic import BaseModel, create_model

from fastapi_mlflow._mlflow_types import (
    build_model_fields,
    MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP,
)


class PyFuncModelPredictError(Exception):
    def __init__(self, exc: Exception):
        super().__init__()
        self.error_type_name = exc.__class__.__name__
        self.message = str(exc)

    def to_dict(self):
        return {"name": self.error_type_name, "message": self.message}


@no_type_check  # Some types here are too dynamic for type checking
def build_predictor(model: PyFuncModel) -> Callable[[BaseModel], Any]:
    """Build and return a function that wraps the mlflow model.

    Currently supports only the `pyfunc`_ flavour of mlflow.

    :param model: PyFuncModel
    :return: Function suitable for mounting as a FastAPI endpoint or route.

    Example::

        model = load_model("/Users/me/path/to/local/model")
        predictor = build_predictor(model)

    .. _pyfunc: https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html

    """
    input_schema = model.metadata.get_input_schema()
    input_model = create_model("RequestRow", **(build_model_fields(input_schema)))
    output_model = create_model(
        "ResponseRow",
        **(build_model_fields(model.metadata.get_output_schema(), nullable=True))
    )

    class Request(BaseModel):
        data: List[input_model]

    class Response(BaseModel):
        data: List[output_model]

    def request_to_dataframe(request: Request) -> pd.DataFrame:
        df = pd.DataFrame([row.dict() for row in request.data], dtype=object)
        for item in input_schema.to_dict():
            if item["type"] in ("integer", "int32"):
                df[item["name"]] = df[item["name"]].astype(np.int32)
            elif item["type"] == "datetime":
                df[item["name"]] = pd.to_datetime(df[item["name"]])
            else:
                df[item["name"]] = df[item["name"]].astype(
                    MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP.get(item["type"], object)
                )

        return df

    def predictor(request: Request) -> Response:
        try:
            predictions = model.predict(request_to_dataframe(request))
        except Exception as exc:
            raise PyFuncModelPredictError(exc) from exc

        response_data = convert_predictions_to_python(predictions)
        return Response(data=response_data)

    return predictor  # type: ignore


def convert_predictions_to_python(results) -> List[Dict[str, Any]]:
    """Convert and return predictions in native Python types."""
    try:
        response_data = (
            results.fillna(np.nan).replace([np.nan], [None]).to_dict(orient="records")
        )
    except (AttributeError, TypeError):
        # Return type is probably a simple array-like
        # Replace NaN with None
        response_data = []
        for row in results:
            value = row if not pd.isnull(row) else None
            response_data.append({"prediction": value})
    return response_data

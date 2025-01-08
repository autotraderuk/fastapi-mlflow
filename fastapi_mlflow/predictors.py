# -*- coding: utf-8 -*-
"""Build functions to wrap mlflow models.

Existing implementation returns a synchronous/blocking function. This decision
was taken because applying ML models is probably CPU-bound. Not having
unnecessary asynchronous code also makes testing simpler.

Current supports only the pyfunc flavour.

Copyright (C) 2022, Auto Trader UK

"""

from typing import Any, Callable, Dict, List, no_type_check

import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel  # type: ignore
from pydantic import BaseModel, create_model

from fastapi_mlflow._mlflow_types import (
    MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP,
    build_model_fields,
)
from fastapi_mlflow.exceptions import DictSerialisableException


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
        **(build_model_fields(model.metadata.get_output_schema(), nullable=True)),
    )

    class Request(BaseModel):
        data: List[input_model]

    class Response(BaseModel):
        data: List[output_model]

    async def request_to_dataframe(request: Request) -> pd.DataFrame:
        df = pd.DataFrame([row.model_dump() for row in request.data], dtype=object)
        for item in input_schema.to_dict():
            if item["type"] in ("integer", "int32"):
                df[item["name"]] = df[item["name"]].astype(np.int32)
            elif item["type"] == "datetime":
                df[item["name"]] = pd.to_datetime(df[item["name"]], utc=True)
            else:
                df[item["name"]] = df[item["name"]].astype(
                    MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP.get(item["type"], object)
                )

        return df

    async def predictor(request: Request) -> Response:
        try:
            predictions = model.predict(await request_to_dataframe(request))
            response_data = await convert_predictions_to_python(predictions)
            return Response(data=response_data)
        except Exception as exc:
            raise DictSerialisableException.from_exception(exc) from exc

    return predictor  # type: ignore


async def convert_predictions_to_python(results) -> List[Dict[str, Any]]:
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

# -*- coding: utf-8 -*-
"""Mapping from mlflow schemas to Pydantic models and native Python types.

Copyright (C) 2022, Auto Trader UK

"""
from mlflow.types import Schema
import pydantic


MLFLOW_SIGNATURE_TO_NUMPY_TYPE_MAP = {
    "double": float,
    "float64": float,
    "int64": int,
    "integer": int,
    "long": int,
    "string": str,
}


def build_input_model(schema: Schema) -> pydantic.BaseModel:
    fields = {
        item["name"]: (
            MLFLOW_SIGNATURE_TO_NUMPY_TYPE_MAP.get(item["type"]),
            ...,  # ... because default value is unknown
        )
        for item in schema.to_dict()
    }
    return pydantic.create_model("Request", **fields)


def build_output_model(schema: Schema):
    rtype = MLFLOW_SIGNATURE_TO_NUMPY_TYPE_MAP.get(
        schema.numpy_types()[0].name
    )
    response_model = pydantic.create_model(
        "Response",
        prediction=(rtype, ...),  # ... because default value is unknown
    )
    return response_model
# -*- coding: utf-8 -*-
"""Mapping from mlflow schemas to Pydantic models and native Python types.

Copyright (C) 2022, Auto Trader UK

"""
from datetime import date, datetime
from typing import Union

from mlflow.types import Schema  # type: ignore

MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP = {
    "boolean": bool,
    "integer": int,
    "long": int,
    "int32": int,
    "int64": int,
    "double": float,
    "float": float,
    "float32": float,
    "float64": float,
    "string": str,
    "binary": bytes,
    "datetime": Union[datetime, date],
}


def build_model_fields(schema):
    fields = {}
    if schema.has_input_names():
        fields.update(
            {
                item["name"]: (
                    MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP.get(item["type"]),
                    ...,  # ... because default value is unknown
                )
                for item in schema.to_dict()
            }
        )
    else:
        rtype = MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP.get(schema.numpy_types()[0].name)
        fields["prediction"] = (rtype, ...)  # ... because default value is unknown

    return fields

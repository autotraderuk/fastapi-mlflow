# -*- coding: utf-8 -*-
"""Mapping from mlflow schemas to Pydantic models and native Python types.

Copyright (C) 2022, Auto Trader UK

"""
from datetime import date, datetime
from typing import Dict, Optional, Union

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


def get_field(type_name: str, nullable: bool):
    """
    :param nullable (bool): Should field be nullable
    """
    type_ = MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP.get(type_name)
    if nullable:
        type_ = Optional[type_]

    field = (type_, ...)  # Ellipsis (...) because default value is unknown
    return field


def build_model_fields(schema, nullable: bool = False) -> Dict:
    """Return a dict mapping field names -> (type, default).

    :param nullable (bool): Should fields be nullable
    """
    if schema.has_input_names():
        return {
            item["name"]: get_field(item["type"], nullable) for item in schema.to_dict()
        }
    else:
        return {"prediction": get_field(schema.numpy_types()[0].name, nullable)}

# -*- coding: utf-8 -*-
"""Mapping from mlflow schemas to Pydantic models and native Python types.

Copyright (C) 2022, Auto Trader UK

"""

from datetime import date, datetime
from typing import Dict, Optional, Union, Tuple

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
    "str": str,
    "object": str,
    "binary": bytes,
    "datetime": Union[datetime, date],
}


class UnsupportedFieldTypeError(Exception):
    pass


def get_field(type_name: str, nullable: bool) -> Tuple:
    """
    :param nullable (bool): Should field be nullable
    """
    try:
        type_ = MLFLOW_SIGNATURE_TO_PYTHON_TYPE_MAP[type_name]
    except KeyError:
        raise UnsupportedFieldTypeError(f"Field type not supported: {type_name}")

    if nullable:
        type_ = Optional[type_]

    field = (type_, ...)  # Ellipsis (...) because default value is unknown
    return field


def build_model_fields(schema, nullable: bool = False) -> Dict[str, Tuple]:
    """Return a dict mapping field names -> (type, default).

    :param nullable (bool): Should fields be nullable
    """
    if schema.has_input_names():
        return {
            item["name"]: get_field(item["type"], nullable) for item in schema.to_dict()
        }
    else:
        return {"prediction": get_field(schema.numpy_types()[0].name, nullable)}

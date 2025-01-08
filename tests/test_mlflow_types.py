# -*- coding: utf-8 -*-
"""Test _mlflow_types.

Copyright (C) 2022, Auto Trader UK
Created 21. Nov 2022

"""

from typing import Optional

import pytest
from mlflow.types.schema import ColSpec, DataType, Schema  # type: ignore

from fastapi_mlflow._mlflow_types import UnsupportedFieldTypeError, build_model_fields


@pytest.fixture
def schema() -> Schema:
    return Schema(
        [
            ColSpec(DataType.integer, "integer"),
            ColSpec(DataType.long, "long"),
            ColSpec(DataType.float, "float"),
            ColSpec(DataType.double, "double"),
            ColSpec(DataType.string, "string"),
        ]
    )


@pytest.fixture
def schema_unnamed() -> Schema:
    return Schema(
        [
            ColSpec(DataType.integer),
        ]
    )


@pytest.fixture
def schema_tensor() -> Schema:
    return Schema.from_json(
        '[{"type": "tensor", "tensor-spec": {"dtype": "object", "shape": [-1]}}]'
    )


def test_build_model_fields(schema: Schema):
    fields = build_model_fields(schema)
    assert 5 == len(fields)
    assert int is fields["integer"][0]
    assert int is fields["long"][0]
    assert float is fields["float"][0]
    assert float is fields["double"][0]
    assert str is fields["string"][0]


def test_build_model_fields_handles_unnamed(schema_unnamed: Schema):
    fields = build_model_fields(schema_unnamed)
    assert 1 == len(fields)
    assert int is fields["prediction"][0]


def test_build_model_fields_handles_nullable(schema: Schema):
    fields = build_model_fields(schema, nullable=True)
    assert 5 == len(fields)
    assert Optional[int] is fields["integer"][0]
    assert Optional[int] is fields["long"][0]
    assert Optional[float] is fields["float"][0]
    assert Optional[float] is fields["double"][0]
    assert Optional[str] is fields["string"][0]


def test_build_model_fields_handles_unnamed_fields_nullable(schema_unnamed: Schema):
    fields = build_model_fields(schema_unnamed, nullable=True)
    assert 1 == len(fields)
    assert Optional[int] is fields["prediction"][0]


def test_build_model_fields_raises_error_on_unknown_type():
    schema = Schema.from_json(
        '[{"type": "tensor", "tensor-spec": {"dtype": "c", "shape": [-1]}}]'
    )
    with pytest.raises(UnsupportedFieldTypeError):
        build_model_fields(schema)


def test_build_model_fields_handles_tensors_of_str(schema_tensor):
    fields = build_model_fields(schema_tensor)
    assert 1 == len(fields)
    assert str is fields["prediction"][0]


def test_build_model_fields_handles_tensors_of_str_nullable(schema_tensor):
    fields = build_model_fields(schema_tensor, nullable=True)
    assert 1 == len(fields)
    assert Optional[str] is fields["prediction"][0]

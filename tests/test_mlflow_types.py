# -*- coding: utf-8 -*-
"""Test _mlflow_types.

Copyright (C) 2022, Auto Trader UK
Created 21. Nov 2022

"""
from typing import Optional

import pytest
from mlflow.types.schema import Schema, ColSpec, DataType  # type: ignore
from fastapi_mlflow._mlflow_types import build_model_fields, UnsupportedFieldTypeError


@pytest.fixture
def schema() -> Schema:
    return Schema(
        [
            ColSpec(DataType.integer, "integer"),
            ColSpec(DataType.long, "long"),
            ColSpec(DataType.float, "float"),
            ColSpec(DataType.double, "double"),
        ]
    )


@pytest.fixture
def schema_unnamed() -> Schema:
    return Schema(
        [
            ColSpec(DataType.integer),
        ]
    )


def test_build_model_fields(schema: Schema):
    fields = build_model_fields(schema)
    assert 4 == len(fields)
    assert int == fields["integer"][0]
    assert int == fields["long"][0]
    assert float == fields["float"][0]
    assert float == fields["double"][0]


def test_build_model_fields_unnamed(schema_unnamed: Schema):
    fields = build_model_fields(schema_unnamed)
    assert 1 == len(fields)
    assert int == fields["prediction"][0]


def test_build_model_fields_nullable(schema: Schema):
    fields = build_model_fields(schema, nullable=True)
    assert 4 == len(fields)
    assert Optional[int] == fields["integer"][0]
    assert Optional[int] == fields["long"][0]
    assert Optional[float] == fields["float"][0]
    assert Optional[float] == fields["double"][0]


def test_build_model_fields_unnamed_nullable(schema_unnamed: Schema):
    fields = build_model_fields(schema_unnamed, nullable=True)
    assert 1 == len(fields)
    assert Optional[int] == fields["prediction"][0]


def test_build_model_fields_raises_error_on_unknown_type():
    schema = Schema.from_json('[{"type": "tensor", "tensor-spec": {"dtype": "c", "shape": [-1]}}]')
    with pytest.raises(UnsupportedFieldTypeError):
        build_model_fields(schema)

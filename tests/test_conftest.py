# -*- coding: utf-8 -*-
"""Meta-testing of test fixtures.

Copyright (C) 2022, Auto Trader UK

"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlflow.pyfunc import PyFuncModel  # type: ignore


class TestNDArrayPyFuncModel:
    def test_pyfunc_model_ndarray_instance(self, pyfunc_model_ndarray: PyFuncModel):
        assert isinstance(pyfunc_model_ndarray, PyFuncModel)

    def test_pyfunc_model_ndarray_predict(
        self,
        pyfunc_model_ndarray: PyFuncModel,
        model_input: pd.DataFrame,
        model_output_ndarray: npt.ArrayLike,
    ):
        """PyFunc model with ndarray return type should predict correct values."""
        assert np.equal(
            model_output_ndarray, pyfunc_model_ndarray.predict(model_input)
        ).all()


class TestSeriesPyFuncModel:
    def test_pyfunc_model_series_instance(self, pyfunc_model_series: PyFuncModel):
        assert isinstance(pyfunc_model_series, PyFuncModel)

    def test_pyfunc_model_series_predict(
        self,
        pyfunc_model_series: PyFuncModel,
        model_input: pd.DataFrame,
        model_output_series: pd.Series,
    ):
        """PyFunc model with Series return type should predict correct values."""

        pd.testing.assert_series_equal(
            model_output_series, pyfunc_model_series.predict(model_input)
        )


class TestDataFramePyFuncModel:
    def test_pyfunc_model_dataframe_instance(self, pyfunc_model_dataframe: PyFuncModel):
        assert isinstance(pyfunc_model_dataframe, PyFuncModel)

    def test_pyfunc_model_dataframe_predict(
        self,
        pyfunc_model_dataframe: PyFuncModel,
        model_input: pd.DataFrame,
        model_output_dataframe: pd.DataFrame,
    ):
        """PyFunc model with DataFrame return type should predict correct values."""

        pd.testing.assert_frame_equal(
            model_output_dataframe, pyfunc_model_dataframe.predict(model_input)
        )


def test_pyfunc_model_signature_inputs(pyfunc_model_ndarray):
    schema = pyfunc_model_ndarray.metadata.get_input_schema()
    schema_dict = schema.to_dict()
    assert schema_dict == [
        {"name": "int32", "type": "integer"},
        {"name": "int64", "type": "long"},
        {"name": "double", "type": "double"},
        {"name": "bool", "type": "boolean"},
        {"name": "bytes", "type": "binary"},
        {"name": "str", "type": "string"},
        {"name": "datetime", "type": "datetime"},
    ]

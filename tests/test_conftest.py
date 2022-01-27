# -*- coding: utf-8 -*-
"""Meta-testing of test fixtures.

Copyright (C) 2022, Auto Trader UK

"""
import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel


def test_pyfunc_model_instance(pyfunc_model: PyFuncModel):
    assert isinstance(pyfunc_model, PyFuncModel)


def test_pyfunc_model_predict(
    pyfunc_model: PyFuncModel,
    model_input: pd.DataFrame,
    model_output: np.array,
):
    assert np.equal(model_output, pyfunc_model.predict(model_input)).all()


def test_pyfunc_model_signature_inputs(pyfunc_model: PyFuncModel):
    schema = pyfunc_model.metadata.get_input_schema()
    schema_dict = schema.to_dict()
    assert schema_dict == [
        {"name": "a", "type": "long"},
        {"name": "b", "type": "double"},
        {"name": "c", "type": "boolean"},
        {"name": "d", "type": "binary"},
        {"name": "e", "type": "string"},
        {"name": "f", "type": "datetime"},
    ]

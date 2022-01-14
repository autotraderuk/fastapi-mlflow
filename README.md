# fastapi mlflow

Deploy mlflow models as JSON APIs using FastAPI with minimal new code.

```python
from fastapi_mlflow.applications import build_app
from mlflow.pyfunc import load_model

model = load_model("/Users/me/path/to/local/model")
app = build_app(model)
```

It should be possible to leverage [FastAPIs Sub Applications](https://fastapi.tiangolo.com/advanced/sub-applications/#sub-applications-mounts) to host multiple models (assuming that they have compatible dependencies...):

```python
from fastapi import FastAPI
from fastapi_mlflow.applications import build_app
from mlflow.pyfunc import load_model

app = FastAPI()

model1 = load_model("/Users/me/path/to/local/model1")
model1_app = build_app(model1)
app.mount("/model1", model1_app)

model2 = load_model("/Users/me/path/to/local/model2")
model2_app = build_app(model2)
app.mount("/model2", model2_app)
```

If you want more control over where and how the prediction is mounted, you can do that too:

```python
from inspect import signature

from fastapi import FastAPI
from fastapi_mlflow.predictors import build_predictor
from mlflow.pyfunc import load_model

model = load_model("/Users/me/path/to/local/model")
predictor = build_predictor(model)
app = FastAPI()
app.add_api_route(
    "/classify",
    predictor,
    response_model=signature(predictor).return_annotation,
    methods=["POST"],
)
```

# fastapi mlflow

Deploy [mlflow](https://www.mlflow.org/) models as JSON APIs using [FastAPI](https://fastapi.tiangolo.com) with minimal new code.

## Installation

```shell
pip install fastapi-mlflow
```

For running the app in production, you will also need an ASGI server, such as [Uvicorn](https://www.uvicorn.org) or [Hypercorn](https://gitlab.com/pgjones/hypercorn).

## Install on Apple Silicon (ARM / M1)

If you experience problems installing on a newer generation Apple silicon based device, [this solution from StackOverflow](https://stackoverflow.com/a/67586301) before retrying install has been found to help.

```shell
brew install openblas gfortran
export OPENBLAS="$(brew --prefix openblas)"
```

## License

Copyright © 2022-23 Auto Trader Group plc.

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Examples

### Simple

#### Create

Create a file `main.py` containing:

```python
from fastapi_mlflow.applications import build_app
from mlflow.pyfunc import load_model

model = load_model("/Users/me/path/to/local/model")
app = build_app(model)
```

#### Run

Run the server with:

```shell
uvicorn main:app
```

#### Check

Open your browser at <http://127.0.0.1:8000/docs>

You should see the automatically generated docs for your model, and be able to test it out using the `Try it out` button in the UI.

### Serve multiple models

It should be possible to host multiple models (assuming that they have compatible dependencies...) by leveraging [FastAPIs Sub Applications](https://fastapi.tiangolo.com/advanced/sub-applications/#sub-applications-mounts):

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

[Run](#run) and [Check](#check) as above.

### Custom routing

If you want more control over where and how the prediction end-point is mounted in your API, you can build the predictor function directly and use it as you need:

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

[Run](#run) and [Check](#check) as above.

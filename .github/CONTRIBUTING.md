# Contributing

## Bugs

Please check [the GitHub issue tracker](https://github.com/autotraderuk/fastapi-mlflow/issues) for existing reports before creating a new one. Please add more detail or examples to an exising report if one exists. If you think you found a new bug, please do create a new issue.

## Feature Requests

Please check [the GitHub issue tracker](https://github.com/autotraderuk/fastapi-mlflow/issues) for existing feature requests before creating a new one. Feel free to add more detail, nuance, or examples to existing requests.

## Pull Requests

Please start a conversation though [the GitHub issue tracker](https://github.com/autotraderuk/fastapi-mlflow/issues) before starting work on a pull requests - it might be that someone is already working on the same thing.

Pull requests for new features, or bug fixes should be accompanied by tests.

Code (including tests) contributions should pass static analysis, typing check, and pass all tests. Typing annotations should be included (except when the code is dealing with types that are only knowable at runtime).

If you set up your dev environment using the guide below, you should be able to run these using:

- `poetry run flake8 fastapi_mlflow tests`
- `poetry run black --check fastapi_mlflow tests`
- `poetry run mypy fastapi_mlflow tests`

## Setting up the development environment

### Pre-requisites

- Python 3.8 or 3.9
- [Poetry](https://python-poetry.org/docs/)

```shell
poetry install --with dev
```

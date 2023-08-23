# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Generic error handling passes correct request and exception arguments 

## [0.6.0] - 2023-08-23
### Changed
- Return an asynchronous function from build_predictor
- Use orjson to serialize responses from app returned by build_app
- Improved error handling in app returned by `build_app`; will catch and format generic `Exceptions`
- Use newer pydantic and fastapi for potential performance improvements

## [0.5.0] - 2023-06-09
### Added
- Support for Python 3.10, 3.11

### Security
- mlflow 2.x
- fastapi 0.96.x

## [0.4.1] - 2023-04-17
### Fixed
- Raise an error for unsupported field types. Previously would silently return a `None` resulting in request time errors.
- Support objects for predictions in array-like structures by mapping them to strings
- Improve support for models that return 1d arrays / tensors of strings
- Catch errors when model result does not conform to expected schema, and format as JSON response

## [0.4.0] - 2023-01-31
### Added
- Catch errors raised by ML model's predict method, format them as JSON, and return as the body of the 500 status response.

## [0.3.2] - 2023-01-13
### Fixed
- Support strings for predictions in array-like structures

## [0.3.1] - 2022-11-24
### Fixed
- Assume that prediction output field(s) may be nullable. It's not uncommon to want ML models to return null or nan values at inference time (e.g. extrapolation outside of training data range).
- Coerce nan values in output to null in JSON.

## [0.3.0] - 2022-10-07
### Added
- Support for multiple model outputs by handling `PyFuncModel`s that return `pandas.Series` or `pandas.DataFrame`

### Fixed
- Explicitly cast DataFrame column types before applying model to handle downcast of 32-bit integers.

## [0.2.0] - 2022-08-12
### Changed
- Return an object in JSON APIs, rather than arrays of objects. This provides better interoperability with JVM based clients using the popular Jackson object mapper.

### Fixed
- Unpin FastAPI dependency to benefit from future upgrades

## [0.1.2] - 2022-03-04
### Added
- Explicit license and Copyright statement
- Link to repository in package metadata

## [0.1.1] - 2022-03-04
### Changed
- Improved documentation in the README for public release

### Fixed
- Typing annotations, checked with mypy 

## [0.1.0] - 2022-01-28
### Added
- Basic functionality to build a predictor callable and an app

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Support for multiple model outputs by handling `PyFuncModel`s that return `pandas.Series` or `pandas.DataFrame`

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

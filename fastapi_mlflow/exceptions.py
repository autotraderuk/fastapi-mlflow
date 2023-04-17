# -*- coding: utf-8 -*-
"""Exceptions.

Copyright (C) 2023, Auto Trader UK
Created 17. Apr 2023

"""


class DictSerialisableException(Exception):
    """An Exception wrapper for formatting."""

    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message

    @classmethod
    def from_exception(cls, exc: Exception):
        return cls(name=exc.__class__.__name__, message=str(exc))

    def to_dict(self):
        return {"name": self.name, "message": self.message}

"""Module containing implementations of secure protocols."""

from __future__ import absolute_import

import inspect

from .aby3 import ABY3
from .pond import Pond
from .pond import TFETensor
from .pond import TFEVariable
from .protocol import Protocol
from .protocol import memoize
from .securenn import SecureNN
from .abyi import ABYI


def get_all_funcs():
    """Assemble public method names from all protocols into a list."""
    all_prot_method_names = set()

    protocols = [Pond, SecureNN, ABY3, ABYI]
    for protocol in protocols:
        members = inspect.getmembers(protocol, predicate=inspect.isfunction)
        all_prot_method_names |= set(
            func_name for func_name, _ in members if not func_name.startswith("_")
        )

    return all_prot_method_names


__all__ = [
    "Protocol",
    "memoize",
    "Pond",
    "SecureNN",
    "ABY3",
    "ABYI",
    "TFEVariable",
    "TFETensor",
]

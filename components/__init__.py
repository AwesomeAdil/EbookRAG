# This file makes the components directory a Python package
from .reader import display_reader
from .search import display_search
from .history import display_history
from .settings import display_settings

__all__ = [
    'display_reader',
    'display_search',
    'display_history',
    'display_settings'
]

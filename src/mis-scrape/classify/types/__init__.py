"""
Detection Types Module
Contains different detection type implementations
"""

from .keyword import Keyword
from .nli import NLI
from .llm import LLM

__all__ = ['Keyword', 'NLI', 'LLM']

# Available detection types
AVAILABLE_TYPES = ['keyword', 'nli', 'llm'] 
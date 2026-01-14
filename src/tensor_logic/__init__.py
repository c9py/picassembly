"""
Tensor Logic - A framework for bridging neural and symbolic AI.

This package provides tools for representing logical operations as tensor
computations, enabling differentiable reasoning that can be integrated with
neural networks.
"""

__version__ = "0.1.0"
__author__ = "PICA PICA Team"
__license__ = "CC BY-NC 4.0"

from tensor_logic.core.predicate import Predicate
from tensor_logic.core.rule import Rule
from tensor_logic.core.tensor_logic import TensorLogic
from tensor_logic.core.knowledge_base import KnowledgeBase

__all__ = [
    "Predicate",
    "Rule",
    "TensorLogic",
    "KnowledgeBase",
    "__version__",
]

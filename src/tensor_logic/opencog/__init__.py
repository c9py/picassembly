"""
OpenCog-style integration patterns for tensor logic.
"""

from typing import Dict, Any, Tuple


class Atom:
    """Base class for OpenCog-style atoms."""
    
    def __init__(self, atom_type: str, name: str, truth_value: float = 0.0):
        self.atom_type = atom_type
        self.name = name
        self.truth_value = truth_value


class ConceptNode(Atom):
    """Represents a concept."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__("ConceptNode", name, **kwargs)


__all__ = ["Atom", "ConceptNode"]

"""
Predicate representation for tensor logic.

Predicates are the building blocks of logical statements. This module provides
tensor-based representations of predicates with support for continuous truth values.
"""

from typing import Optional, List, Union, Tuple
import torch
import numpy as np


class Predicate:
    """
    Represents a logical predicate with tensor-based truth values.
    
    A predicate P(x1, x2, ..., xn) has:
    - name: string identifier
    - arity: number of arguments
    - values: tensor storing truth values (0-1 for fuzzy logic)
    
    Examples:
        >>> human = Predicate("human", arity=1, domain_size=100)
        >>> parent = Predicate("parent", arity=2, domain_size=100)
        >>> is_red = Predicate("is_red", arity=1, domain_size=50)
    """
    
    def __init__(
        self,
        name: str,
        arity: int = 1,
        domain_size: Optional[int] = None,
        values: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ):
        """
        Initialize a predicate.
        
        Args:
            name: Name of the predicate
            arity: Number of arguments (0 for propositions)
            domain_size: Size of each domain dimension
            values: Optional pre-initialized tensor values
            device: Torch device ('cpu' or 'cuda')
        """
        self.name = name
        self.arity = arity
        self.domain_size = domain_size or 1
        self.device = device
        
        if values is not None:
            self.values = values.to(device)
        else:
            # Initialize with zeros (unknown/false)
            shape = tuple([self.domain_size] * arity) if arity > 0 else tuple()
            self.values = torch.zeros(shape, dtype=torch.float32, device=device)
    
    def __call__(self, *args: Union[int, str, torch.Tensor]) -> torch.Tensor:
        """
        Query or set predicate truth value.
        
        Args:
            *args: Indices or entity identifiers
            
        Returns:
            Truth value tensor (0-1)
        """
        if len(args) != self.arity:
            raise ValueError(
                f"Expected {self.arity} arguments, got {len(args)}"
            )
        
        if self.arity == 0:
            return self.values
        
        # Convert string args to indices if needed
        indices = []
        for arg in args:
            if isinstance(arg, str):
                # Hash string to index (simplified)
                indices.append(hash(arg) % self.domain_size)
            elif isinstance(arg, int):
                indices.append(arg)
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")
        
        return self.values[tuple(indices)]
    
    def set(self, *args: Union[int, str], value: float = 1.0) -> None:
        """
        Set truth value for specific predicate instantiation.
        
        Args:
            *args: Indices or entity identifiers
            value: Truth value (0-1)
        """
        if len(args) != self.arity:
            raise ValueError(
                f"Expected {self.arity} arguments, got {len(args)}"
            )
        
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Truth value must be in [0, 1], got {value}")
        
        if self.arity == 0:
            self.values = torch.tensor(value, device=self.device)
        else:
            indices = []
            for arg in args:
                if isinstance(arg, str):
                    indices.append(hash(arg) % self.domain_size)
                elif isinstance(arg, int):
                    indices.append(arg)
                else:
                    raise TypeError(f"Unsupported argument type: {type(arg)}")
            
            self.values[tuple(indices)] = value
    
    def get_all_true(self, threshold: float = 0.5) -> List[Tuple[int, ...]]:
        """
        Get all predicate instantiations that are true.
        
        Args:
            threshold: Minimum truth value to consider true
            
        Returns:
            List of index tuples for true instantiations
        """
        if self.arity == 0:
            return [] if self.values.item() < threshold else [()]
        
        mask = self.values >= threshold
        indices = torch.nonzero(mask, as_tuple=False)
        return [tuple(idx.tolist()) for idx in indices]
    
    def clone(self) -> "Predicate":
        """Create a deep copy of this predicate."""
        return Predicate(
            name=self.name,
            arity=self.arity,
            domain_size=self.domain_size,
            values=self.values.clone(),
            device=self.device
        )
    
    def to(self, device: str) -> "Predicate":
        """Move predicate to specified device."""
        self.device = device
        self.values = self.values.to(device)
        return self
    
    def __repr__(self) -> str:
        return f"Predicate('{self.name}', arity={self.arity}, domain_size={self.domain_size})"
    
    def __str__(self) -> str:
        if self.arity == 0:
            return f"{self.name} = {self.values.item():.3f}"
        
        true_count = (self.values >= 0.5).sum().item()
        total = self.values.numel()
        return f"{self.name}/{self.arity} ({true_count}/{total} true)"


class PredicateFactory:
    """Factory for creating and managing predicates."""
    
    def __init__(self, domain_size: int = 100, device: str = "cpu"):
        self.domain_size = domain_size
        self.device = device
        self._predicates = {}
    
    def create(self, name: str, arity: int = 1) -> Predicate:
        """Create a new predicate."""
        key = (name, arity)
        if key not in self._predicates:
            self._predicates[key] = Predicate(
                name=name,
                arity=arity,
                domain_size=self.domain_size,
                device=self.device
            )
        return self._predicates[key]
    
    def get(self, name: str, arity: int = 1) -> Optional[Predicate]:
        """Get existing predicate."""
        return self._predicates.get((name, arity))
    
    def list_all(self) -> List[Predicate]:
        """List all created predicates."""
        return list(self._predicates.values())

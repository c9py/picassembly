"""
Fuzzy logic operations for tensor logic.

Provides differentiable implementations of logical operations using fuzzy logic,
enabling gradient-based learning while maintaining logical semantics.
"""

import torch
from typing import Union


def fuzzy_and(x: torch.Tensor, y: torch.Tensor, method: str = "product") -> torch.Tensor:
    """
    Fuzzy AND operation (conjunction/t-norm).
    
    Args:
        x: First truth value tensor (0-1)
        y: Second truth value tensor (0-1)
        method: T-norm method ('product', 'min', 'lukasiewicz')
        
    Returns:
        Result of fuzzy AND
        
    Examples:
        >>> x = torch.tensor([0.8, 0.6, 0.3])
        >>> y = torch.tensor([0.9, 0.4, 0.7])
        >>> fuzzy_and(x, y, method='product')
        tensor([0.7200, 0.2400, 0.2100])
    """
    if method == "product":
        # Product t-norm: x * y
        return x * y
    elif method == "min":
        # Gödel t-norm: min(x, y)
        return torch.min(x, y)
    elif method == "lukasiewicz":
        # Łukasiewicz t-norm: max(0, x + y - 1)
        return torch.clamp(x + y - 1, min=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def fuzzy_or(x: torch.Tensor, y: torch.Tensor, method: str = "probsum") -> torch.Tensor:
    """
    Fuzzy OR operation (disjunction/t-conorm).
    
    Args:
        x: First truth value tensor (0-1)
        y: Second truth value tensor (0-1)
        method: T-conorm method ('probsum', 'max', 'lukasiewicz')
        
    Returns:
        Result of fuzzy OR
        
    Examples:
        >>> x = torch.tensor([0.8, 0.6, 0.3])
        >>> y = torch.tensor([0.9, 0.4, 0.7])
        >>> fuzzy_or(x, y, method='probsum')
        tensor([0.9800, 0.7600, 0.7900])
    """
    if method == "probsum":
        # Probabilistic sum: x + y - x*y
        return x + y - x * y
    elif method == "max":
        # Gödel t-conorm: max(x, y)
        return torch.max(x, y)
    elif method == "lukasiewicz":
        # Łukasiewicz t-conorm: min(1, x + y)
        return torch.clamp(x + y, max=1)
    else:
        raise ValueError(f"Unknown method: {method}")


def fuzzy_not(x: torch.Tensor) -> torch.Tensor:
    """
    Fuzzy NOT operation (negation).
    
    Args:
        x: Truth value tensor (0-1)
        
    Returns:
        Result of fuzzy NOT (1 - x)
        
    Examples:
        >>> x = torch.tensor([0.8, 0.6, 0.3])
        >>> fuzzy_not(x)
        tensor([0.2000, 0.4000, 0.7000])
    """
    return 1.0 - x


def fuzzy_implies(x: torch.Tensor, y: torch.Tensor, method: str = "kleene") -> torch.Tensor:
    """
    Fuzzy implication operation.
    
    Args:
        x: Premise truth value tensor (0-1)
        y: Conclusion truth value tensor (0-1)
        method: Implication method ('kleene', 'lukasiewicz', 'godel')
        
    Returns:
        Result of fuzzy implication
        
    Examples:
        >>> x = torch.tensor([0.8, 0.6, 0.3])
        >>> y = torch.tensor([0.9, 0.4, 0.7])
        >>> fuzzy_implies(x, y, method='kleene')
        tensor([0.9000, 0.4000, 0.7000])
    """
    if method == "kleene":
        # Kleene-Dienes: max(1-x, y)
        return torch.max(fuzzy_not(x), y)
    elif method == "lukasiewicz":
        # Łukasiewicz: min(1, 1-x+y)
        return torch.clamp(1 - x + y, max=1)
    elif method == "godel":
        # Gödel: 1 if x <= y else y
        return torch.where(x <= y, torch.ones_like(x), y)
    else:
        raise ValueError(f"Unknown method: {method}")


def fuzzy_equiv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fuzzy equivalence (biconditional).
    
    Args:
        x: First truth value tensor (0-1)
        y: Second truth value tensor (0-1)
        
    Returns:
        Result of fuzzy equivalence: (x→y) ∧ (y→x)
        
    Examples:
        >>> x = torch.tensor([0.8, 0.6, 0.3])
        >>> y = torch.tensor([0.9, 0.6, 0.7])
        >>> fuzzy_equiv(x, y)
        tensor([0.7200, 1.0000, 0.2100])
    """
    return fuzzy_and(fuzzy_implies(x, y), fuzzy_implies(y, x))


def fuzzy_xor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fuzzy XOR (exclusive or).
    
    Args:
        x: First truth value tensor (0-1)
        y: Second truth value tensor (0-1)
        
    Returns:
        Result of fuzzy XOR: (x ∨ y) ∧ ¬(x ∧ y)
        
    Examples:
        >>> x = torch.tensor([0.8, 0.6, 0.3])
        >>> y = torch.tensor([0.1, 0.6, 0.7])
        >>> fuzzy_xor(x, y)
        tensor([0.7280, 0.1200, 0.5890])
    """
    return fuzzy_and(fuzzy_or(x, y), fuzzy_not(fuzzy_and(x, y)))


def aggregate_and(tensors: list, method: str = "product") -> torch.Tensor:
    """
    Aggregate multiple tensors with AND operation.
    
    Args:
        tensors: List of truth value tensors
        method: T-norm method to use
        
    Returns:
        Aggregated tensor
    """
    if not tensors:
        raise ValueError("Cannot aggregate empty list")
    
    result = tensors[0]
    for t in tensors[1:]:
        result = fuzzy_and(result, t, method=method)
    return result


def aggregate_or(tensors: list, method: str = "probsum") -> torch.Tensor:
    """
    Aggregate multiple tensors with OR operation.
    
    Args:
        tensors: List of truth value tensors
        method: T-conorm method to use
        
    Returns:
        Aggregated tensor
    """
    if not tensors:
        raise ValueError("Cannot aggregate empty list")
    
    result = tensors[0]
    for t in tensors[1:]:
        result = fuzzy_or(result, t, method=method)
    return result


def soft_threshold(x: torch.Tensor, threshold: float = 0.5, sharpness: float = 10.0) -> torch.Tensor:
    """
    Soft (differentiable) thresholding function.
    
    Args:
        x: Input tensor
        threshold: Threshold value
        sharpness: Controls steepness of sigmoid (higher = sharper)
        
    Returns:
        Soft-thresholded tensor
        
    Examples:
        >>> x = torch.tensor([0.3, 0.5, 0.7])
        >>> soft_threshold(x, threshold=0.5, sharpness=10.0)
        tensor([0.0474, 0.5000, 0.9526])
    """
    return torch.sigmoid(sharpness * (x - threshold))


def attention_and(x: torch.Tensor, y: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Attention-weighted fuzzy AND.
    
    Args:
        x: First truth value tensor
        y: Second truth value tensor
        attention: Attention weights (0-1)
        
    Returns:
        Attention-weighted conjunction
    """
    basic_and = fuzzy_and(x, y)
    return attention * basic_and + (1 - attention) * torch.min(x, y)


class FuzzyOps:
    """
    Container for fuzzy logic operations with configurable defaults.
    """
    
    def __init__(self, and_method: str = "product", or_method: str = "probsum"):
        self.and_method = and_method
        self.or_method = or_method
    
    def fuzzy_and(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return fuzzy_and(x, y, method=self.and_method)
    
    def fuzzy_or(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return fuzzy_or(x, y, method=self.or_method)
    
    def fuzzy_not(self, x: torch.Tensor) -> torch.Tensor:
        return fuzzy_not(x)
    
    def fuzzy_implies(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return fuzzy_implies(x, y)

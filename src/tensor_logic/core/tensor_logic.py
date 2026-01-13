"""
Main TensorLogic engine.

Provides the primary interface for tensor-based logical reasoning.
"""

from typing import Dict, List, Optional, Union
import torch
from tensor_logic.core.predicate import Predicate
from tensor_logic.core.rule import Rule
from tensor_logic.core.knowledge_base import KnowledgeBase
from tensor_logic.ops import FuzzyOps


class TensorLogic:
    """
    Main tensor logic engine for neural-symbolic reasoning.
    
    Combines:
    - Knowledge representation (predicates, rules, facts)
    - Fuzzy logic operations (continuous truth values)
    - Inference mechanisms (forward/backward chaining)
    - Neural integration (differentiable operations)
    
    Examples:
        >>> tl = TensorLogic(domain_size=100)
        >>> 
        >>> # Define predicates
        >>> human = tl.predicate("human", arity=1)
        >>> mortal = tl.predicate("mortal", arity=1)
        >>> 
        >>> # Add rule
        >>> tl.add_rule(human, mortal, confidence=1.0)
        >>> 
        >>> # Add fact
        >>> tl.fact(human, "socrates", value=1.0)
        >>> 
        >>> # Infer and query
        >>> tl.infer()
        >>> result = tl.query(mortal, "socrates")
        >>> print(f"mortal(socrates) = {result:.3f}")
    """
    
    def __init__(
        self,
        domain_size: int = 100,
        device: str = "cpu",
        and_method: str = "product",
        or_method: str = "probsum"
    ):
        """
        Initialize tensor logic engine.
        
        Args:
            domain_size: Default domain size for predicates
            device: Torch device ('cpu' or 'cuda')
            and_method: Fuzzy AND method ('product', 'min', 'lukasiewicz')
            or_method: Fuzzy OR method ('probsum', 'max', 'lukasiewicz')
        """
        self.kb = KnowledgeBase(domain_size, device)
        self.fuzzy_ops = FuzzyOps(and_method, or_method)
        self.device = device
    
    def predicate(self, name: str, arity: int = 1) -> Predicate:
        """
        Create or get a predicate.
        
        Args:
            name: Predicate name
            arity: Number of arguments
            
        Returns:
            Predicate instance
        """
        return self.kb.add_predicate(name, arity)
    
    def add_rule(
        self,
        premise: Union[Predicate, List[Predicate]],
        conclusion: Predicate,
        confidence: float = 1.0
    ) -> Rule:
        """
        Add a logical rule.
        
        Args:
            premise: Premise predicate(s)
            conclusion: Conclusion predicate
            confidence: Rule confidence (0-1)
            
        Returns:
            Created rule
        """
        rule = Rule(premise, conclusion, confidence)
        self.kb.rules.add(rule)
        return rule
    
    def fact(self, predicate: Predicate, *args, value: float = 1.0) -> None:
        """
        Assert a fact.
        
        Args:
            predicate: Predicate to assert
            *args: Entity identifiers
            value: Truth value (0-1)
        """
        predicate.set(*args, value=value)
    
    def query(self, predicate: Predicate, *args) -> float:
        """
        Query truth value of a predicate instantiation.
        
        Args:
            predicate: Predicate to query
            *args: Entity identifiers
            
        Returns:
            Truth value (0-1)
        """
        result = predicate(*args)
        return result.item() if result.numel() == 1 else result
    
    def infer(self, max_iterations: int = 100) -> int:
        """
        Run inference to derive new facts.
        
        Args:
            max_iterations: Maximum iterations
            
        Returns:
            Number of iterations performed
        """
        return self.kb.infer(max_iterations)
    
    def clear(self) -> None:
        """Clear all knowledge."""
        self.kb.clear()
    
    def to(self, device: str) -> "TensorLogic":
        """Move to specified device."""
        self.device = device
        self.kb.to(device)
        return self
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        return self.kb.summary()
    
    def __repr__(self) -> str:
        return f"TensorLogic({self.kb})"

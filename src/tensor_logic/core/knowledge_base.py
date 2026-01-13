"""
Knowledge Base for tensor logic.

Manages predicates, rules, and facts in a unified knowledge representation.
"""

from typing import Dict, List, Optional, Set, Tuple
import torch
from tensor_logic.core.predicate import Predicate, PredicateFactory
from tensor_logic.core.rule import Rule, RuleSet


class KnowledgeBase:
    """
    Knowledge base for storing and querying logical knowledge.
    
    Manages:
    - Predicates (definitions and truth values)
    - Rules (implications and transformations)
    - Facts (ground truth statements)
    
    Examples:
        >>> kb = KnowledgeBase(domain_size=100)
        >>> kb.add_predicate("human", arity=1)
        >>> kb.add_predicate("mortal", arity=1)
        >>> kb.add_rule("human", "mortal")
        >>> kb.set_fact("human", "socrates", value=1.0)
        >>> result = kb.query("mortal", "socrates")
    """
    
    def __init__(self, domain_size: int = 100, device: str = "cpu"):
        """
        Initialize knowledge base.
        
        Args:
            domain_size: Default domain size for predicates
            device: Torch device ('cpu' or 'cuda')
        """
        self.domain_size = domain_size
        self.device = device
        self.predicate_factory = PredicateFactory(domain_size, device)
        self.rules = RuleSet()
        self._entity_index: Dict[str, int] = {}
        self._next_entity_id = 0
    
    def add_predicate(self, name: str, arity: int = 1) -> Predicate:
        """
        Add a new predicate to the knowledge base.
        
        Args:
            name: Predicate name
            arity: Number of arguments
            
        Returns:
            Created predicate
        """
        return self.predicate_factory.create(name, arity)
    
    def get_predicate(self, name: str, arity: int = 1) -> Optional[Predicate]:
        """Get predicate by name and arity."""
        return self.predicate_factory.get(name, arity)
    
    def add_rule(
        self,
        premise: str,
        conclusion: str,
        confidence: float = 1.0,
        premise_arity: int = 1,
        conclusion_arity: int = 1
    ) -> Rule:
        """
        Add a rule to the knowledge base.
        
        Args:
            premise: Name of premise predicate
            conclusion: Name of conclusion predicate
            confidence: Rule confidence (0-1)
            premise_arity: Arity of premise predicate
            conclusion_arity: Arity of conclusion predicate
            
        Returns:
            Created rule
        """
        premise_pred = self.get_predicate(premise, premise_arity)
        conclusion_pred = self.get_predicate(conclusion, conclusion_arity)
        
        if premise_pred is None:
            premise_pred = self.add_predicate(premise, premise_arity)
        if conclusion_pred is None:
            conclusion_pred = self.add_predicate(conclusion, conclusion_arity)
        
        rule = Rule(
            premise=premise_pred,
            conclusion=conclusion_pred,
            confidence=confidence
        )
        self.rules.add(rule)
        return rule
    
    def set_fact(self, predicate_name: str, *args, value: float = 1.0, arity: int = None) -> None:
        """
        Set a fact (ground truth) in the knowledge base.
        
        Args:
            predicate_name: Name of the predicate
            *args: Entity identifiers
            value: Truth value (0-1)
            arity: Arity of predicate (inferred from args if None)
        """
        if arity is None:
            arity = len(args)
        
        pred = self.get_predicate(predicate_name, arity)
        if pred is None:
            pred = self.add_predicate(predicate_name, arity)
        
        # Convert string entities to indices
        indices = []
        for arg in args:
            if isinstance(arg, str):
                if arg not in self._entity_index:
                    self._entity_index[arg] = self._next_entity_id
                    self._next_entity_id += 1
                indices.append(self._entity_index[arg])
            else:
                indices.append(arg)
        
        pred.set(*indices, value=value)
    
    def query(self, predicate_name: str, *args, arity: int = None) -> float:
        """
        Query a fact from the knowledge base.
        
        Args:
            predicate_name: Name of the predicate
            *args: Entity identifiers
            arity: Arity of predicate (inferred from args if None)
            
        Returns:
            Truth value (0-1)
        """
        if arity is None:
            arity = len(args)
        
        pred = self.get_predicate(predicate_name, arity)
        if pred is None:
            return 0.0
        
        # Convert string entities to indices
        indices = []
        for arg in args:
            if isinstance(arg, str):
                if arg not in self._entity_index:
                    return 0.0  # Unknown entity
                indices.append(self._entity_index[arg])
            else:
                indices.append(arg)
        
        result = pred(*indices)
        return result.item() if result.numel() == 1 else result
    
    def infer(self, max_iterations: int = 100) -> int:
        """
        Run inference to derive new facts from rules.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Number of iterations performed
        """
        return self.rules.forward_chain_all(max_iterations)
    
    def get_all_facts(self, threshold: float = 0.5) -> List[Tuple[str, Tuple, float]]:
        """
        Get all facts in the knowledge base.
        
        Args:
            threshold: Minimum truth value
            
        Returns:
            List of (predicate_name, args, truth_value) tuples
        """
        facts = []
        for pred in self.predicate_factory.list_all():
            true_instances = pred.get_all_true(threshold)
            for instance in true_instances:
                value = pred(*instance).item()
                facts.append((pred.name, instance, value))
        return facts
    
    def clear(self) -> None:
        """Clear all predicates, rules, and facts."""
        self.predicate_factory = PredicateFactory(self.domain_size, self.device)
        self.rules = RuleSet()
        self._entity_index = {}
        self._next_entity_id = 0
    
    def to(self, device: str) -> "KnowledgeBase":
        """Move knowledge base to specified device."""
        self.device = device
        for pred in self.predicate_factory.list_all():
            pred.to(device)
        return self
    
    def summary(self) -> Dict:
        """Get summary statistics of the knowledge base."""
        predicates = self.predicate_factory.list_all()
        num_facts = sum(
            (p.values >= 0.5).sum().item() for p in predicates
        )
        
        return {
            "num_predicates": len(predicates),
            "num_rules": len(self.rules),
            "num_facts": num_facts,
            "num_entities": len(self._entity_index),
            "device": self.device
        }
    
    def __repr__(self) -> str:
        stats = self.summary()
        return (
            f"KnowledgeBase("
            f"predicates={stats['num_predicates']}, "
            f"rules={stats['num_rules']}, "
            f"facts={stats['num_facts']})"
        )

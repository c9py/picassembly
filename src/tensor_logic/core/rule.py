"""
Rule representation for tensor logic.

Rules define logical implications and transformations between predicates.
"""

from typing import Optional, List, Union, Callable
import torch
from tensor_logic.core.predicate import Predicate


class Rule:
    """
    Represents a logical rule (implication).
    
    A rule has the form: premise -> conclusion
    or more complex forms: P1 ∧ P2 ∧ ... ∧ Pn -> Q
    
    Examples:
        >>> # Simple rule: human(X) -> mortal(X)
        >>> rule1 = Rule(premise=human, conclusion=mortal)
        >>> 
        >>> # Composite rule: parent(X,Y) ∧ parent(Y,Z) -> grandparent(X,Z)
        >>> rule2 = Rule(
        ...     premise=[parent, parent],
        ...     conclusion=grandparent,
        ...     aggregation="and"
        ... )
    """
    
    def __init__(
        self,
        premise: Union[Predicate, List[Predicate]],
        conclusion: Predicate,
        confidence: float = 1.0,
        aggregation: str = "and",
        transformation: Optional[Callable] = None
    ):
        """
        Initialize a rule.
        
        Args:
            premise: Single predicate or list of predicates in the premise
            conclusion: Conclusion predicate
            confidence: Rule confidence/strength (0-1)
            aggregation: How to combine multiple premises ('and', 'or', 'custom')
            transformation: Optional custom transformation function
        """
        self.premise = [premise] if isinstance(premise, Predicate) else premise
        self.conclusion = conclusion
        self.confidence = confidence
        self.aggregation = aggregation
        self.transformation = transformation
        
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")
    
    def apply(self, fuzzy_ops=None) -> torch.Tensor:
        """
        Apply the rule to derive conclusion truth values.
        
        Args:
            fuzzy_ops: Optional fuzzy logic operations module
            
        Returns:
            Truth values for the conclusion
        """
        if fuzzy_ops is None:
            from tensor_logic.ops import FuzzyOps
            fuzzy_ops = FuzzyOps()
        
        # Get premise truth values
        premise_values = [p.values for p in self.premise]
        
        # Aggregate premises
        if len(premise_values) == 1:
            aggregated = premise_values[0]
        elif self.aggregation == "and":
            aggregated = premise_values[0]
            for pv in premise_values[1:]:
                aggregated = fuzzy_ops.fuzzy_and(aggregated, pv)
        elif self.aggregation == "or":
            aggregated = premise_values[0]
            for pv in premise_values[1:]:
                aggregated = fuzzy_ops.fuzzy_or(aggregated, pv)
        elif self.aggregation == "custom" and self.transformation:
            aggregated = self.transformation(*premise_values)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Apply confidence
        result = aggregated * self.confidence
        
        return result
    
    def forward_chain(self) -> None:
        """
        Apply forward chaining: if premise is true, make conclusion true.
        """
        result = self.apply()
        
        # Update conclusion with max of existing and derived values
        self.conclusion.values = torch.max(
            self.conclusion.values,
            result
        )
    
    def get_premise_names(self) -> List[str]:
        """Get names of all premise predicates."""
        return [p.name for p in self.premise]
    
    def get_conclusion_name(self) -> str:
        """Get name of conclusion predicate."""
        return self.conclusion.name
    
    def __repr__(self) -> str:
        premise_str = " ∧ ".join(p.name for p in self.premise)
        return f"Rule({premise_str} → {self.conclusion.name}, conf={self.confidence:.2f})"
    
    def __str__(self) -> str:
        premise_str = " ∧ ".join(p.name for p in self.premise)
        return f"{premise_str} → {self.conclusion.name}"


class RuleSet:
    """
    Collection of rules for organized rule management.
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
    
    def add(self, rule: Rule) -> None:
        """Add a rule to the set."""
        self.rules.append(rule)
    
    def remove(self, rule: Rule) -> None:
        """Remove a rule from the set."""
        self.rules.remove(rule)
    
    def get_rules_for_predicate(self, predicate_name: str) -> List[Rule]:
        """Get all rules that conclude the given predicate."""
        return [
            rule for rule in self.rules
            if rule.get_conclusion_name() == predicate_name
        ]
    
    def get_rules_using_predicate(self, predicate_name: str) -> List[Rule]:
        """Get all rules that use the given predicate in their premise."""
        return [
            rule for rule in self.rules
            if predicate_name in rule.get_premise_names()
        ]
    
    def forward_chain_all(self, max_iterations: int = 100) -> int:
        """
        Apply forward chaining to all rules until convergence.
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Number of iterations performed
        """
        for iteration in range(max_iterations):
            changed = False
            
            for rule in self.rules:
                old_values = rule.conclusion.values.clone()
                rule.forward_chain()
                
                # Check if conclusion changed significantly
                if torch.any(torch.abs(rule.conclusion.values - old_values) > 1e-6):
                    changed = True
            
            if not changed:
                return iteration + 1
        
        return max_iterations
    
    def __len__(self) -> int:
        return len(self.rules)
    
    def __iter__(self):
        return iter(self.rules)
    
    def __repr__(self) -> str:
        return f"RuleSet({len(self.rules)} rules)"

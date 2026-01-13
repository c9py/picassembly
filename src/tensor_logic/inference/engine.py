"""
Inference algorithms for tensor logic.

Implements various reasoning strategies including forward chaining,
backward chaining, and probabilistic inference.
"""

from typing import List, Set, Optional, Tuple
import torch
from tensor_logic.core.predicate import Predicate
from tensor_logic.core.rule import Rule, RuleSet


class InferenceEngine:
    """
    Base inference engine for logical reasoning.
    """
    
    def __init__(self, rules: RuleSet):
        self.rules = rules
    
    def forward_chain(self, max_iterations: int = 100) -> int:
        """
        Forward chaining (bottom-up) inference.
        
        Derives new facts from existing facts and rules until
        no new facts can be derived or max iterations reached.
        
        Args:
            max_iterations: Maximum iterations
            
        Returns:
            Number of iterations performed
        """
        return self.rules.forward_chain_all(max_iterations)


class BackwardChaining:
    """
    Backward chaining (top-down) inference.
    
    Given a goal, works backward to find supporting facts and rules.
    """
    
    def __init__(self, rules: RuleSet):
        self.rules = rules
    
    def prove(
        self,
        goal: Predicate,
        max_depth: int = 10,
        visited: Optional[Set[str]] = None
    ) -> Tuple[bool, float]:
        """
        Prove a goal using backward chaining.
        
        Args:
            goal: Goal predicate to prove
            max_depth: Maximum recursion depth
            visited: Set of visited predicates (for cycle detection)
            
        Returns:
            Tuple of (success, confidence)
        """
        if visited is None:
            visited = set()
        
        if max_depth <= 0:
            return False, 0.0
        
        # Prevent cycles
        if goal.name in visited:
            return False, 0.0
        
        visited.add(goal.name)
        
        # Check if goal already has truth value
        if (goal.values >= 0.5).any():
            max_val = goal.values.max().item()
            return True, max_val
        
        # Find rules that conclude the goal
        relevant_rules = self.rules.get_rules_for_predicate(goal.name)
        
        if not relevant_rules:
            return False, 0.0
        
        # Try to prove premises of each rule
        best_confidence = 0.0
        for rule in relevant_rules:
            premise_confidence = 1.0
            
            for premise_pred in rule.premise:
                success, conf = self.prove(
                    premise_pred,
                    max_depth - 1,
                    visited.copy()
                )
                if not success:
                    premise_confidence = 0.0
                    break
                premise_confidence = min(premise_confidence, conf)
            
            if premise_confidence > 0:
                rule_confidence = premise_confidence * rule.confidence
                best_confidence = max(best_confidence, rule_confidence)
        
        return best_confidence > 0, best_confidence


class ProbabilisticInference:
    """
    Probabilistic inference with uncertainty propagation.
    """
    
    def __init__(self, rules: RuleSet):
        self.rules = rules
    
    def infer_with_uncertainty(
        self,
        iterations: int = 10,
        noise_std: float = 0.01
    ) -> None:
        """
        Perform probabilistic inference with Monte Carlo sampling.
        
        Args:
            iterations: Number of MC samples
            noise_std: Standard deviation of noise to add
        """
        # Store original values
        original_values = {}
        for rule in self.rules:
            for pred in rule.premise:
                if pred.name not in original_values:
                    original_values[pred.name] = pred.values.clone()
        
        # Run multiple inference iterations with noise
        accumulated_results = {}
        
        for _ in range(iterations):
            # Add noise to premise values
            for name, values in original_values.items():
                # Find corresponding predicate
                for rule in self.rules:
                    for pred in rule.premise:
                        if pred.name == name:
                            noise = torch.randn_like(values) * noise_std
                            pred.values = torch.clamp(values + noise, 0, 1)
            
            # Run inference
            self.rules.forward_chain_all(max_iterations=10)
            
            # Accumulate results
            for rule in self.rules:
                conclusion_name = rule.conclusion.name
                if conclusion_name not in accumulated_results:
                    accumulated_results[conclusion_name] = []
                accumulated_results[conclusion_name].append(
                    rule.conclusion.values.clone()
                )
        
        # Average results
        for conclusion_name, values_list in accumulated_results.items():
            mean_values = torch.stack(values_list).mean(dim=0)
            # Update conclusion with mean
            for rule in self.rules:
                if rule.conclusion.name == conclusion_name:
                    rule.conclusion.values = mean_values
        
        # Restore original premise values
        for name, values in original_values.items():
            for rule in self.rules:
                for pred in rule.premise:
                    if pred.name == name:
                        pred.values = values


class AbductiveReasoning:
    """
    Abductive reasoning - inference to the best explanation.
    """
    
    def __init__(self, rules: RuleSet):
        self.rules = rules
    
    def explain(
        self,
        observation: Predicate,
        threshold: float = 0.5
    ) -> List[Tuple[Rule, float]]:
        """
        Find best explanations for an observation.
        
        Args:
            observation: Observed predicate
            threshold: Minimum confidence for explanations
            
        Returns:
            List of (rule, explanation_score) tuples
        """
        explanations = []
        
        # Find rules that could explain the observation
        relevant_rules = self.rules.get_rules_for_predicate(observation.name)
        
        for rule in relevant_rules:
            # Calculate how well the premises explain the observation
            premise_values = [p.values for p in rule.premise]
            
            if not premise_values:
                continue
            
            # Calculate joint premise probability
            joint_prob = premise_values[0]
            for pv in premise_values[1:]:
                joint_prob = joint_prob * pv
            
            # Calculate explanation score
            expected_conclusion = joint_prob * rule.confidence
            actual_conclusion = observation.values
            
            # Similarity between expected and actual
            similarity = 1.0 - torch.abs(expected_conclusion - actual_conclusion).mean()
            score = similarity.item()
            
            if score >= threshold:
                explanations.append((rule, score))
        
        # Sort by score (best explanations first)
        explanations.sort(key=lambda x: x[1], reverse=True)
        
        return explanations

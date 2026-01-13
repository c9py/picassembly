"""
Unit tests for Rule and RuleSet classes.
"""

import pytest
import torch
from tensor_logic.core.predicate import Predicate
from tensor_logic.core.rule import Rule, RuleSet


class TestRule:
    """Test suite for Rule class."""
    
    def test_rule_creation(self):
        """Test basic rule creation."""
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        
        rule = Rule(premise=human, conclusion=mortal, confidence=1.0)
        
        assert len(rule.premise) == 1
        assert rule.premise[0] is human
        assert rule.conclusion is mortal
        assert rule.confidence == 1.0
    
    def test_rule_with_multiple_premises(self):
        """Test rule with multiple premises."""
        parent = Predicate("parent", arity=2, domain_size=10)
        grandparent = Predicate("grandparent", arity=2, domain_size=10)
        
        rule = Rule(
            premise=[parent, parent],
            conclusion=grandparent,
            confidence=0.9
        )
        
        assert len(rule.premise) == 2
        assert rule.confidence == 0.9
    
    def test_rule_forward_chain(self):
        """Test forward chaining with a simple rule."""
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        
        # Set some humans
        human.set(0, value=1.0)
        human.set(1, value=0.8)
        
        # Create rule: human -> mortal
        rule = Rule(premise=human, conclusion=mortal, confidence=1.0)
        rule.forward_chain()
        
        # Mortal should now have values
        assert mortal(0).item() == 1.0
        assert mortal(1).item() == 0.8
        assert mortal(5).item() == 0.0
    
    def test_rule_with_confidence(self):
        """Test rule confidence scaling."""
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        
        human.set(0, value=1.0)
        
        # Rule with 70% confidence
        rule = Rule(premise=human, conclusion=mortal, confidence=0.7)
        rule.forward_chain()
        
        # Conclusion should be scaled by confidence
        assert mortal(0).item() == 0.7
    
    def test_rule_aggregation_and(self):
        """Test rule with AND aggregation."""
        p1 = Predicate("p1", arity=1, domain_size=10)
        p2 = Predicate("p2", arity=1, domain_size=10)
        conclusion = Predicate("conclusion", arity=1, domain_size=10)
        
        p1.set(0, value=0.8)
        p2.set(0, value=0.6)
        
        rule = Rule(
            premise=[p1, p2],
            conclusion=conclusion,
            aggregation="and"
        )
        rule.forward_chain()
        
        # AND with product: 0.8 * 0.6 = 0.48
        assert abs(conclusion(0).item() - 0.48) < 0.01
    
    def test_invalid_confidence(self):
        """Test error handling for invalid confidence."""
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        
        with pytest.raises(ValueError):
            Rule(premise=human, conclusion=mortal, confidence=1.5)
        
        with pytest.raises(ValueError):
            Rule(premise=human, conclusion=mortal, confidence=-0.5)


class TestRuleSet:
    """Test suite for RuleSet class."""
    
    def test_ruleset_creation(self):
        """Test creating a rule set."""
        ruleset = RuleSet()
        assert len(ruleset) == 0
    
    def test_add_remove_rules(self):
        """Test adding and removing rules."""
        ruleset = RuleSet()
        
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        
        rule = Rule(premise=human, conclusion=mortal)
        
        ruleset.add(rule)
        assert len(ruleset) == 1
        
        ruleset.remove(rule)
        assert len(ruleset) == 0
    
    def test_get_rules_for_predicate(self):
        """Test finding rules that conclude a predicate."""
        ruleset = RuleSet()
        
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        wise = Predicate("wise", arity=1, domain_size=10)
        
        rule1 = Rule(premise=human, conclusion=mortal)
        rule2 = Rule(premise=wise, conclusion=mortal)
        rule3 = Rule(premise=human, conclusion=wise)
        
        ruleset.add(rule1)
        ruleset.add(rule2)
        ruleset.add(rule3)
        
        mortal_rules = ruleset.get_rules_for_predicate("mortal")
        assert len(mortal_rules) == 2
        assert rule1 in mortal_rules
        assert rule2 in mortal_rules
    
    def test_get_rules_using_predicate(self):
        """Test finding rules that use a predicate in premise."""
        ruleset = RuleSet()
        
        human = Predicate("human", arity=1, domain_size=10)
        mortal = Predicate("mortal", arity=1, domain_size=10)
        wise = Predicate("wise", arity=1, domain_size=10)
        
        rule1 = Rule(premise=human, conclusion=mortal)
        rule2 = Rule(premise=human, conclusion=wise)
        rule3 = Rule(premise=wise, conclusion=mortal)
        
        ruleset.add(rule1)
        ruleset.add(rule2)
        ruleset.add(rule3)
        
        human_rules = ruleset.get_rules_using_predicate("human")
        assert len(human_rules) == 2
        assert rule1 in human_rules
        assert rule2 in human_rules
    
    def test_forward_chain_all(self):
        """Test forward chaining on entire rule set."""
        ruleset = RuleSet()
        
        # Create chain: a -> b -> c
        a = Predicate("a", arity=1, domain_size=10)
        b = Predicate("b", arity=1, domain_size=10)
        c = Predicate("c", arity=1, domain_size=10)
        
        a.set(0, value=1.0)
        
        rule1 = Rule(premise=a, conclusion=b)
        rule2 = Rule(premise=b, conclusion=c)
        
        ruleset.add(rule1)
        ruleset.add(rule2)
        
        iterations = ruleset.forward_chain_all(max_iterations=10)
        
        # Should propagate through chain
        assert b(0).item() == 1.0
        assert c(0).item() == 1.0
        assert iterations <= 3
    
    def test_forward_chain_convergence(self):
        """Test that forward chaining converges."""
        ruleset = RuleSet()
        
        # Create reflexive rule: a -> a (with confidence < 1)
        a = Predicate("a", arity=1, domain_size=10)
        a.set(0, value=0.5)
        
        rule = Rule(premise=a, conclusion=a, confidence=0.9)
        ruleset.add(rule)
        
        iterations = ruleset.forward_chain_all(max_iterations=100)
        
        # Should converge quickly
        assert iterations < 100

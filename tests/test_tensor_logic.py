"""
Unit tests for TensorLogic main engine.
"""

import pytest
import torch
from tensor_logic import TensorLogic, Predicate, Rule


class TestTensorLogic:
    """Test suite for TensorLogic class."""
    
    def test_tensorlogic_creation(self):
        """Test basic TensorLogic creation."""
        tl = TensorLogic(domain_size=100)
        
        assert tl.kb.domain_size == 100
        assert tl.device == "cpu"
    
    def test_predicate_creation(self):
        """Test creating predicates through TensorLogic."""
        tl = TensorLogic()
        
        human = tl.predicate("human", arity=1)
        mortal = tl.predicate("mortal", arity=1)
        
        assert human.name == "human"
        assert mortal.name == "mortal"
    
    def test_add_rule(self):
        """Test adding rules through TensorLogic."""
        tl = TensorLogic()
        
        human = tl.predicate("human", arity=1)
        mortal = tl.predicate("mortal", arity=1)
        
        rule = tl.add_rule(human, mortal, confidence=1.0)
        
        assert len(tl.kb.rules) == 1
        assert rule.confidence == 1.0
    
    def test_fact_assertion(self):
        """Test asserting facts through TensorLogic."""
        tl = TensorLogic()
        
        human = tl.predicate("human", arity=1)
        
        tl.fact(human, "socrates", value=1.0)
        
        result = tl.query(human, "socrates")
        assert result == 1.0
    
    def test_simple_inference(self):
        """Test simple inference scenario."""
        tl = TensorLogic(domain_size=100)
        
        # Define predicates
        human = tl.predicate("human", arity=1)
        mortal = tl.predicate("mortal", arity=1)
        
        # Add rule: human -> mortal
        tl.add_rule(human, mortal, confidence=1.0)
        
        # Add fact: human(socrates)
        tl.fact(human, "socrates", value=1.0)
        
        # Infer
        tl.infer()
        
        # Query: mortal(socrates)?
        result = tl.query(mortal, "socrates")
        assert result == 1.0
    
    def test_chained_inference(self):
        """Test chained reasoning."""
        tl = TensorLogic(domain_size=100)
        
        # Define predicates
        human = tl.predicate("human", arity=1)
        mortal = tl.predicate("mortal", arity=1)
        finite = tl.predicate("finite", arity=1)
        
        # Add rules
        tl.add_rule(human, mortal)
        tl.add_rule(mortal, finite)
        
        # Add fact
        tl.fact(human, "socrates", value=1.0)
        
        # Infer
        tl.infer()
        
        # Check chain propagation
        assert tl.query(mortal, "socrates") == 1.0
        assert tl.query(finite, "socrates") == 1.0
    
    def test_multiple_premises(self):
        """Test rules with multiple premises."""
        tl = TensorLogic(domain_size=100)
        
        p1 = tl.predicate("p1", arity=1)
        p2 = tl.predicate("p2", arity=1)
        conclusion = tl.predicate("conclusion", arity=1)
        
        tl.add_rule([p1, p2], conclusion)
        
        tl.fact(p1, "x", value=0.8)
        tl.fact(p2, "x", value=0.6)
        
        tl.infer()
        
        result = tl.query(conclusion, "x")
        # AND with product: 0.8 * 0.6 = 0.48
        assert abs(result - 0.48) < 0.01
    
    def test_clear(self):
        """Test clearing TensorLogic."""
        tl = TensorLogic()
        
        human = tl.predicate("human", arity=1)
        tl.fact(human, "socrates", value=1.0)
        
        tl.clear()
        
        assert len(tl.kb.predicate_factory.list_all()) == 0
    
    def test_summary(self):
        """Test TensorLogic summary."""
        tl = TensorLogic()
        
        human = tl.predicate("human", arity=1)
        mortal = tl.predicate("mortal", arity=1)
        tl.add_rule(human, mortal)
        tl.fact(human, "socrates", value=1.0)
        
        summary = tl.summary()
        
        assert summary["num_predicates"] == 2
        assert summary["num_rules"] == 1
    
    def test_fuzzy_inference(self):
        """Test inference with fuzzy truth values."""
        tl = TensorLogic()
        
        human = tl.predicate("human", arity=1)
        mortal = tl.predicate("mortal", arity=1)
        
        tl.add_rule(human, mortal, confidence=0.9)
        
        # Fuzzy fact: 80% sure someone is human
        tl.fact(human, "x", value=0.8)
        
        tl.infer()
        
        # Should propagate with both fuzzy values
        result = tl.query(mortal, "x")
        assert abs(result - 0.72) < 0.01  # 0.8 * 0.9
    
    def test_device_support(self):
        """Test device support (CPU/CUDA)."""
        tl = TensorLogic(device="cpu")
        
        human = tl.predicate("human", arity=1)
        assert human.device == "cpu"
        
        # Test moving to different device (would need CUDA to fully test)
        tl_moved = tl.to("cpu")
        assert tl_moved.device == "cpu"

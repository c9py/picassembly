"""
Unit tests for Predicate class.
"""

import pytest
import torch
from tensor_logic.core.predicate import Predicate, PredicateFactory


class TestPredicate:
    """Test suite for Predicate class."""
    
    def test_predicate_creation(self):
        """Test basic predicate creation."""
        pred = Predicate("human", arity=1, domain_size=10)
        assert pred.name == "human"
        assert pred.arity == 1
        assert pred.domain_size == 10
        assert pred.values.shape == (10,)
    
    def test_predicate_zero_arity(self):
        """Test proposition (0-arity predicate)."""
        prop = Predicate("raining", arity=0)
        assert prop.arity == 0
        assert prop.values.shape == torch.Size([])
    
    def test_predicate_binary(self):
        """Test binary predicate (arity=2)."""
        pred = Predicate("parent", arity=2, domain_size=5)
        assert pred.arity == 2
        assert pred.values.shape == (5, 5)
    
    def test_set_and_get_value(self):
        """Test setting and getting predicate values."""
        pred = Predicate("human", arity=1, domain_size=10)
        pred.set(0, value=1.0)
        pred.set(5, value=0.7)
        
        assert pred(0).item() == 1.0
        assert pred(5).item() == 0.7
        assert pred(3).item() == 0.0
    
    def test_set_with_string_args(self):
        """Test setting values with string arguments."""
        pred = Predicate("human", arity=1, domain_size=100)
        pred.set("socrates", value=1.0)
        pred.set("plato", value=0.9)
        
        # Should be deterministic based on hash
        val1 = pred("socrates").item()
        val2 = pred("plato").item()
        
        assert val1 == 1.0
        assert val2 == 0.9
    
    def test_binary_predicate_operations(self):
        """Test binary predicate operations."""
        pred = Predicate("parent", arity=2, domain_size=10)
        pred.set(0, 1, value=1.0)
        pred.set(0, 2, value=1.0)
        pred.set(1, 3, value=0.8)
        
        assert pred(0, 1).item() == 1.0
        assert pred(0, 2).item() == 1.0
        assert pred(1, 3).item() == 0.8
        assert pred(5, 5).item() == 0.0
    
    def test_get_all_true(self):
        """Test getting all true instances."""
        pred = Predicate("test", arity=1, domain_size=10)
        pred.set(0, value=1.0)
        pred.set(3, value=0.8)
        pred.set(7, value=0.6)
        
        true_instances = pred.get_all_true(threshold=0.5)
        assert len(true_instances) == 3
        assert (0,) in true_instances
        assert (3,) in true_instances
        assert (7,) in true_instances
    
    def test_clone(self):
        """Test cloning predicates."""
        pred1 = Predicate("test", arity=1, domain_size=5)
        pred1.set(0, value=1.0)
        pred1.set(2, value=0.5)
        
        pred2 = pred1.clone()
        
        assert pred2.name == pred1.name
        assert pred2.arity == pred1.arity
        assert torch.equal(pred2.values, pred1.values)
        
        # Modify clone shouldn't affect original
        pred2.set(0, value=0.0)
        assert pred1(0).item() == 1.0
        assert pred2(0).item() == 0.0
    
    def test_invalid_arity(self):
        """Test error handling for wrong number of arguments."""
        pred = Predicate("test", arity=2, domain_size=5)
        
        with pytest.raises(ValueError):
            pred(0)  # Should require 2 arguments
        
        with pytest.raises(ValueError):
            pred.set(0, value=1.0)  # Should require 2 arguments
    
    def test_invalid_truth_value(self):
        """Test error handling for invalid truth values."""
        pred = Predicate("test", arity=1, domain_size=5)
        
        with pytest.raises(ValueError):
            pred.set(0, value=-0.5)
        
        with pytest.raises(ValueError):
            pred.set(0, value=1.5)


class TestPredicateFactory:
    """Test suite for PredicateFactory class."""
    
    def test_factory_creation(self):
        """Test creating predicates with factory."""
        factory = PredicateFactory(domain_size=50)
        
        human = factory.create("human", arity=1)
        mortal = factory.create("mortal", arity=1)
        
        assert human.domain_size == 50
        assert mortal.domain_size == 50
    
    def test_factory_get(self):
        """Test retrieving predicates from factory."""
        factory = PredicateFactory(domain_size=50)
        
        human1 = factory.create("human", arity=1)
        human2 = factory.get("human", arity=1)
        
        assert human1 is human2
    
    def test_factory_list_all(self):
        """Test listing all predicates."""
        factory = PredicateFactory(domain_size=50)
        
        factory.create("human", arity=1)
        factory.create("mortal", arity=1)
        factory.create("parent", arity=2)
        
        all_preds = factory.list_all()
        assert len(all_preds) == 3

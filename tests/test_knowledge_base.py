"""
Unit tests for KnowledgeBase class.
"""

import pytest
import torch
from tensor_logic.core.knowledge_base import KnowledgeBase


class TestKnowledgeBase:
    """Test suite for KnowledgeBase class."""
    
    def test_kb_creation(self):
        """Test basic knowledge base creation."""
        kb = KnowledgeBase(domain_size=100)
        
        assert kb.domain_size == 100
        assert len(kb.predicate_factory.list_all()) == 0
        assert len(kb.rules) == 0
    
    def test_add_predicate(self):
        """Test adding predicates to KB."""
        kb = KnowledgeBase()
        
        human = kb.add_predicate("human", arity=1)
        mortal = kb.add_predicate("mortal", arity=1)
        
        assert human.name == "human"
        assert mortal.name == "mortal"
        assert len(kb.predicate_factory.list_all()) == 2
    
    def test_get_predicate(self):
        """Test retrieving predicates from KB."""
        kb = KnowledgeBase()
        
        kb.add_predicate("human", arity=1)
        
        retrieved = kb.get_predicate("human", arity=1)
        assert retrieved is not None
        assert retrieved.name == "human"
        
        missing = kb.get_predicate("nonexistent", arity=1)
        assert missing is None
    
    def test_add_rule(self):
        """Test adding rules to KB."""
        kb = KnowledgeBase()
        
        rule = kb.add_rule("human", "mortal", confidence=1.0)
        
        assert len(kb.rules) == 1
        assert rule.confidence == 1.0
    
    def test_set_fact(self):
        """Test setting facts in KB."""
        kb = KnowledgeBase()
        
        kb.add_predicate("human", arity=1)
        kb.set_fact("human", "socrates", value=1.0)
        
        # Should create entity mapping
        assert "socrates" in kb._entity_index
    
    def test_query_fact(self):
        """Test querying facts from KB."""
        kb = KnowledgeBase()
        
        kb.add_predicate("human", arity=1)
        kb.set_fact("human", "socrates", value=1.0)
        kb.set_fact("human", "plato", value=0.9)
        
        assert abs(kb.query("human", "socrates") - 1.0) < 0.01
        assert abs(kb.query("human", "plato") - 0.9) < 0.01
        assert kb.query("human", "unknown") == 0.0
    
    def test_inference(self):
        """Test inference in KB."""
        kb = KnowledgeBase(domain_size=100)
        
        # Add predicates
        kb.add_predicate("human", arity=1)
        kb.add_predicate("mortal", arity=1)
        
        # Add rule: human -> mortal
        kb.add_rule("human", "mortal", confidence=1.0)
        
        # Add fact: human(socrates)
        kb.set_fact("human", "socrates", value=1.0)
        
        # Run inference
        iterations = kb.infer(max_iterations=10)
        
        # Should infer: mortal(socrates)
        assert kb.query("mortal", "socrates") == 1.0
        assert iterations <= 2
    
    def test_chain_inference(self):
        """Test chained inference."""
        kb = KnowledgeBase(domain_size=100)
        
        # Create chain: a -> b -> c
        kb.add_predicate("a", arity=1)
        kb.add_predicate("b", arity=1)
        kb.add_predicate("c", arity=1)
        
        kb.add_rule("a", "b")
        kb.add_rule("b", "c")
        
        kb.set_fact("a", "x", value=1.0)
        
        kb.infer()
        
        assert kb.query("b", "x") == 1.0
        assert kb.query("c", "x") == 1.0
    
    def test_get_all_facts(self):
        """Test retrieving all facts."""
        kb = KnowledgeBase(domain_size=100)
        
        kb.add_predicate("human", arity=1)
        kb.set_fact("human", "socrates", value=1.0)
        kb.set_fact("human", "plato", value=0.9)
        kb.set_fact("human", "aristotle", value=0.8)
        
        facts = kb.get_all_facts(threshold=0.5)
        
        assert len(facts) >= 3
        fact_names = [f[0] for f in facts]
        assert "human" in fact_names
    
    def test_clear_kb(self):
        """Test clearing knowledge base."""
        kb = KnowledgeBase()
        
        kb.add_predicate("human", arity=1)
        kb.add_rule("human", "mortal")
        kb.set_fact("human", "socrates", value=1.0)
        
        kb.clear()
        
        assert len(kb.predicate_factory.list_all()) == 0
        assert len(kb.rules) == 0
        assert len(kb._entity_index) == 0
    
    def test_kb_summary(self):
        """Test knowledge base summary."""
        kb = KnowledgeBase()
        
        kb.add_predicate("human", arity=1)
        kb.add_predicate("mortal", arity=1)
        kb.add_rule("human", "mortal")
        kb.set_fact("human", "socrates", value=1.0)
        
        summary = kb.summary()
        
        assert summary["num_predicates"] == 2
        assert summary["num_rules"] == 1
        assert summary["num_entities"] == 1
        assert "device" in summary
    
    def test_binary_predicate_facts(self):
        """Test facts with binary predicates."""
        kb = KnowledgeBase()
        
        kb.add_predicate("parent", arity=2)
        kb.set_fact("parent", "john", "mary", value=1.0)
        kb.set_fact("parent", "mary", "susan", value=1.0)
        
        assert kb.query("parent", "john", "mary") == 1.0
        assert kb.query("parent", "mary", "susan") == 1.0
        assert kb.query("parent", "john", "susan") == 0.0

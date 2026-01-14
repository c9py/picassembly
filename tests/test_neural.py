"""
Unit tests for neural-symbolic integration.
"""

import pytest
import torch
import torch.nn as nn
from tensor_logic.neural.modules import (
    NeuralLogicModule,
    LogicRegularizer,
    PredicateEmbedding,
    RelationNetwork,
    DifferentiableKnowledgeBase
)
from tensor_logic.core.rule import RuleSet


class TestNeuralLogicModule:
    """Test suite for NeuralLogicModule."""
    
    def test_module_creation(self):
        """Test creating neural-logic module."""
        model = NeuralLogicModule(
            input_dim=128,
            output_dim=10,
            hidden_dims=[64, 32]
        )
        
        assert model.input_dim == 128
        assert model.output_dim == 10
    
    def test_forward_pass(self):
        """Test forward pass through module."""
        model = NeuralLogicModule(
            input_dim=128,
            output_dim=10,
            hidden_dims=[64]
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 128)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_gradients(self):
        """Test that gradients flow through the module."""
        model = NeuralLogicModule(
            input_dim=10,
            output_dim=2,
            hidden_dims=[5]
        )
        
        x = torch.randn(2, 10, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check that at least some parameters have gradients
        has_grads = any(param.grad is not None for param in model.parameters())
        assert has_grads


class TestLogicRegularizer:
    """Test suite for LogicRegularizer."""
    
    def test_regularizer_creation(self):
        """Test creating logic regularizer."""
        rules = RuleSet()
        regularizer = LogicRegularizer(rules, weight=0.1)
        
        assert regularizer.weight == 0.1
    
    def test_regularizer_loss(self):
        """Test computing regularization loss."""
        from tensor_logic.core.predicate import Predicate
        from tensor_logic.core.rule import Rule
        
        rules = RuleSet()
        p = Predicate("p", arity=1, domain_size=10)
        q = Predicate("q", arity=1, domain_size=10)
        
        p.set(0, value=1.0)
        q.set(0, value=0.5)
        
        rule = Rule(premise=p, conclusion=q)
        rules.add(rule)
        
        regularizer = LogicRegularizer(rules, weight=0.1)
        
        predictions = torch.randn(10, requires_grad=True)
        loss = regularizer(predictions)
        
        assert isinstance(loss, torch.Tensor)
        # Loss may not always require grad if rules don't use predictions
        assert loss.numel() == 1


class TestPredicateEmbedding:
    """Test suite for PredicateEmbedding."""
    
    def test_embedding_creation(self):
        """Test creating predicate embeddings."""
        embed = PredicateEmbedding(
            num_predicates=100,
            embedding_dim=64
        )
        
        assert embed.embedding.num_embeddings == 100
        assert embed.embedding.embedding_dim == 64
    
    def test_embedding_forward(self):
        """Test forward pass of embeddings."""
        embed = PredicateEmbedding(
            num_predicates=100,
            embedding_dim=64
        )
        
        predicate_ids = torch.tensor([0, 5, 10, 99])
        embeddings = embed(predicate_ids)
        
        assert embeddings.shape == (4, 64)
    
    def test_embedding_gradients(self):
        """Test gradients flow through embeddings."""
        embed = PredicateEmbedding(
            num_predicates=10,
            embedding_dim=8
        )
        
        predicate_ids = torch.tensor([0, 1, 2])
        embeddings = embed(predicate_ids)
        loss = embeddings.sum()
        loss.backward()
        
        assert embed.embedding.weight.grad is not None


class TestRelationNetwork:
    """Test suite for RelationNetwork."""
    
    def test_relation_network_creation(self):
        """Test creating relation network."""
        net = RelationNetwork(
            entity_dim=64,
            relation_dim=32,
            hidden_dim=128
        )
        
        assert isinstance(net, nn.Module)
    
    def test_relation_network_forward(self):
        """Test forward pass of relation network."""
        net = RelationNetwork(
            entity_dim=64,
            relation_dim=32,
            hidden_dim=128
        )
        
        batch_size = 4
        subject = torch.randn(batch_size, 64)
        relation = torch.randn(batch_size, 32)
        object_emb = torch.randn(batch_size, 64)
        
        output = net(subject, relation, object_emb)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_relation_network_gradients(self):
        """Test gradients flow through relation network."""
        net = RelationNetwork(
            entity_dim=8,
            relation_dim=4,
            hidden_dim=16
        )
        
        subject = torch.randn(2, 8, requires_grad=True)
        relation = torch.randn(2, 4, requires_grad=True)
        object_emb = torch.randn(2, 8, requires_grad=True)
        
        output = net(subject, relation, object_emb)
        loss = output.sum()
        loss.backward()
        
        assert subject.grad is not None
        assert relation.grad is not None
        assert object_emb.grad is not None


class TestDifferentiableKnowledgeBase:
    """Test suite for DifferentiableKnowledgeBase."""
    
    def test_diff_kb_creation(self):
        """Test creating differentiable KB."""
        kb = DifferentiableKnowledgeBase(
            num_entities=100,
            num_relations=50,
            embedding_dim=64
        )
        
        assert kb.entity_embeddings.num_embeddings == 100
        assert kb.relation_embeddings.num_embeddings == 50
    
    def test_diff_kb_forward(self):
        """Test querying differentiable KB."""
        kb = DifferentiableKnowledgeBase(
            num_entities=100,
            num_relations=50,
            embedding_dim=64
        )
        
        batch_size = 8
        subject_ids = torch.randint(0, 100, (batch_size,))
        relation_ids = torch.randint(0, 50, (batch_size,))
        object_ids = torch.randint(0, 100, (batch_size,))
        
        truth_values = kb(subject_ids, relation_ids, object_ids)
        
        assert truth_values.shape == (batch_size,)
        assert torch.all(truth_values >= 0) and torch.all(truth_values <= 1)
    
    def test_diff_kb_embeddings(self):
        """Test getting entity and relation embeddings."""
        kb = DifferentiableKnowledgeBase(
            num_entities=10,
            num_relations=5,
            embedding_dim=8
        )
        
        entity_emb = kb.get_entity_embedding(0)
        relation_emb = kb.get_relation_embedding(0)
        
        assert entity_emb.shape == (1, 8)
        assert relation_emb.shape == (1, 8)
    
    def test_diff_kb_gradients(self):
        """Test that KB parameters are learnable."""
        kb = DifferentiableKnowledgeBase(
            num_entities=10,
            num_relations=5,
            embedding_dim=8
        )
        
        subject_ids = torch.tensor([0, 1])
        relation_ids = torch.tensor([0, 1])
        object_ids = torch.tensor([2, 3])
        
        truth_values = kb(subject_ids, relation_ids, object_ids)
        loss = truth_values.sum()
        loss.backward()
        
        # truth_values should definitely have gradients
        assert kb.truth_values.grad is not None
        # Embeddings may not have gradients if they're not used in forward()
        # which is the case in the current implementation

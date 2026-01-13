"""
Unit tests for fuzzy logic operations.
"""

import pytest
import torch
from tensor_logic.ops import (
    fuzzy_and,
    fuzzy_or,
    fuzzy_not,
    fuzzy_implies,
    fuzzy_equiv,
    fuzzy_xor,
    aggregate_and,
    aggregate_or,
    soft_threshold,
    FuzzyOps
)


class TestFuzzyLogicOperations:
    """Test suite for fuzzy logic operations."""
    
    def test_fuzzy_and_product(self):
        """Test fuzzy AND with product t-norm."""
        x = torch.tensor([0.8, 0.6, 0.3])
        y = torch.tensor([0.9, 0.4, 0.7])
        
        result = fuzzy_and(x, y, method="product")
        
        expected = torch.tensor([0.72, 0.24, 0.21])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_and_min(self):
        """Test fuzzy AND with min t-norm."""
        x = torch.tensor([0.8, 0.6, 0.3])
        y = torch.tensor([0.9, 0.4, 0.7])
        
        result = fuzzy_and(x, y, method="min")
        
        expected = torch.tensor([0.8, 0.4, 0.3])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_and_lukasiewicz(self):
        """Test fuzzy AND with Lukasiewicz t-norm."""
        x = torch.tensor([0.8, 0.6, 0.3])
        y = torch.tensor([0.9, 0.4, 0.7])
        
        result = fuzzy_and(x, y, method="lukasiewicz")
        
        expected = torch.tensor([0.7, 0.0, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_or_probsum(self):
        """Test fuzzy OR with probabilistic sum."""
        x = torch.tensor([0.8, 0.6, 0.3])
        y = torch.tensor([0.9, 0.4, 0.7])
        
        result = fuzzy_or(x, y, method="probsum")
        
        # x + y - x*y
        expected = torch.tensor([0.98, 0.76, 0.79])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_or_max(self):
        """Test fuzzy OR with max t-conorm."""
        x = torch.tensor([0.8, 0.6, 0.3])
        y = torch.tensor([0.9, 0.4, 0.7])
        
        result = fuzzy_or(x, y, method="max")
        
        expected = torch.tensor([0.9, 0.6, 0.7])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_not(self):
        """Test fuzzy NOT operation."""
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        
        result = fuzzy_not(x)
        
        expected = torch.tensor([1.0, 0.7, 0.5, 0.3, 0.0])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_implies_kleene(self):
        """Test fuzzy implication with Kleene-Dienes."""
        x = torch.tensor([0.8, 0.6, 0.3])
        y = torch.tensor([0.9, 0.4, 0.7])
        
        result = fuzzy_implies(x, y, method="kleene")
        
        # max(1-x, y)
        expected = torch.tensor([0.9, 0.4, 0.7])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_fuzzy_equiv(self):
        """Test fuzzy equivalence."""
        x = torch.tensor([0.8, 0.5, 0.3])
        y = torch.tensor([0.8, 0.5, 0.7])
        
        result = fuzzy_equiv(x, y)
        
        # High when x and y are similar
        assert result[0].item() > 0.8
        assert result[1].item() > 0.8
        assert result[2].item() < 0.5
    
    def test_fuzzy_xor(self):
        """Test fuzzy XOR."""
        x = torch.tensor([1.0, 0.0, 0.5])
        y = torch.tensor([0.0, 1.0, 0.5])
        
        result = fuzzy_xor(x, y)
        
        # XOR is high when inputs differ
        assert result[0].item() > 0.5
        assert result[1].item() > 0.5
        assert result[2].item() < 0.5
    
    def test_aggregate_and(self):
        """Test aggregating multiple tensors with AND."""
        t1 = torch.tensor([0.9, 0.8, 0.7])
        t2 = torch.tensor([0.8, 0.7, 0.6])
        t3 = torch.tensor([0.7, 0.6, 0.5])
        
        result = aggregate_and([t1, t2, t3], method="product")
        
        expected = t1 * t2 * t3
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_aggregate_or(self):
        """Test aggregating multiple tensors with OR."""
        t1 = torch.tensor([0.5, 0.3, 0.2])
        t2 = torch.tensor([0.4, 0.2, 0.1])
        t3 = torch.tensor([0.3, 0.1, 0.0])
        
        result = aggregate_or([t1, t2, t3], method="max")
        
        expected = torch.tensor([0.5, 0.3, 0.2])
        assert torch.allclose(result, expected, atol=1e-5)
    
    def test_soft_threshold(self):
        """Test soft thresholding."""
        x = torch.tensor([0.3, 0.5, 0.7])
        
        result = soft_threshold(x, threshold=0.5, sharpness=10.0)
        
        # Values below threshold should be close to 0
        # Values above threshold should be close to 1
        assert result[0].item() < 0.3
        assert 0.4 < result[1].item() < 0.6
        assert result[2].item() > 0.7
    
    def test_gradients(self):
        """Test that operations are differentiable."""
        x = torch.tensor([0.8], requires_grad=True)
        y = torch.tensor([0.6], requires_grad=True)
        
        z = fuzzy_and(x, y, method="product")
        z.backward()
        
        assert x.grad is not None
        assert y.grad is not None
        assert x.grad.item() > 0
        assert y.grad.item() > 0


class TestFuzzyOps:
    """Test FuzzyOps container class."""
    
    def test_fuzzy_ops_defaults(self):
        """Test FuzzyOps with default settings."""
        ops = FuzzyOps()
        
        x = torch.tensor([0.8])
        y = torch.tensor([0.6])
        
        result_and = ops.fuzzy_and(x, y)
        result_or = ops.fuzzy_or(x, y)
        result_not = ops.fuzzy_not(x)
        
        assert result_and.item() == 0.48  # product
        assert result_or.item() > 0.9  # probsum
        assert result_not.item() == 0.2
    
    def test_fuzzy_ops_custom_methods(self):
        """Test FuzzyOps with custom methods."""
        ops = FuzzyOps(and_method="min", or_method="max")
        
        x = torch.tensor([0.8])
        y = torch.tensor([0.6])
        
        result_and = ops.fuzzy_and(x, y)
        result_or = ops.fuzzy_or(x, y)
        
        assert result_and.item() == 0.6  # min
        assert result_or.item() == 0.8  # max

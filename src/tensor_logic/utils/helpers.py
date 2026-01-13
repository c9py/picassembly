"""
Utility functions for tensor logic.
"""

import torch
from typing import List, Dict, Any
import json


def tensor_to_dict(tensor: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Convert tensor to dictionary representation.
    
    Args:
        tensor: Input tensor
        threshold: Threshold for binary conversion
        
    Returns:
        Dictionary representation
    """
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "num_true": (tensor >= threshold).sum().item(),
        "num_false": (tensor < threshold).sum().item(),
    }


def visualize_predicate(predicate, threshold: float = 0.5) -> str:
    """
    Create text visualization of predicate truth values.
    
    Args:
        predicate: Predicate to visualize
        threshold: Threshold for considering true
        
    Returns:
        String visualization
    """
    lines = [f"Predicate: {predicate.name}/{predicate.arity}"]
    lines.append(f"Domain size: {predicate.domain_size}")
    
    if predicate.arity == 0:
        val = predicate.values.item()
        lines.append(f"Value: {val:.3f} {'(TRUE)' if val >= threshold else '(FALSE)'}")
    else:
        true_instances = predicate.get_all_true(threshold)
        lines.append(f"True instances: {len(true_instances)}/{predicate.values.numel()}")
        
        if len(true_instances) <= 10:
            for inst in true_instances:
                val = predicate(*inst).item()
                lines.append(f"  {inst}: {val:.3f}")
        else:
            for inst in true_instances[:5]:
                val = predicate(*inst).item()
                lines.append(f"  {inst}: {val:.3f}")
            lines.append(f"  ... ({len(true_instances) - 5} more)")
    
    return "\n".join(lines)


def save_knowledge_base(kb, filepath: str) -> None:
    """
    Save knowledge base to file.
    
    Args:
        kb: KnowledgeBase instance
        filepath: Path to save file
    """
    data = {
        "domain_size": kb.domain_size,
        "entity_index": kb._entity_index,
        "predicates": {},
        "rules": []
    }
    
    # Save predicates
    for pred in kb.predicate_factory.list_all():
        data["predicates"][f"{pred.name}_{pred.arity}"] = {
            "name": pred.name,
            "arity": pred.arity,
            "values": pred.values.cpu().numpy().tolist()
        }
    
    # Save rules (simplified)
    for rule in kb.rules:
        data["rules"].append({
            "premise": rule.get_premise_names(),
            "conclusion": rule.get_conclusion_name(),
            "confidence": rule.confidence
        })
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def similarity_score(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute similarity between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        
    Returns:
        Similarity score (0-1)
    """
    # Cosine similarity
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()
    
    dot_product = torch.dot(flat1, flat2)
    norm1 = torch.norm(flat1)
    norm2 = torch.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return (dot_product / (norm1 * norm2)).item()

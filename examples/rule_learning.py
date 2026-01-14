"""
Rule learning example.

Demonstrates automatic discovery and learning of logical rules from data.
"""

import torch
from tensor_logic import TensorLogic
from tensor_logic.core.predicate import Predicate


def discover_rules(tl, threshold=0.7):
    """
    Simple rule discovery based on correlation.
    
    Discovers potential rules by finding predicates that frequently
    co-occur in the knowledge base.
    """
    discovered = []
    predicates = tl.kb.predicate_factory.list_all()
    
    for i, pred1 in enumerate(predicates):
        for pred2 in predicates[i+1:]:
            # Skip if different arities
            if pred1.arity != pred2.arity:
                continue
            
            # Calculate correlation between predicates
            true1 = (pred1.values >= 0.5).float()
            true2 = (pred2.values >= 0.5).float()
            
            # Correlation: (both_true) / (pred1_true)
            both_true = (true1 * true2).sum()
            pred1_true = true1.sum()
            
            if pred1_true > 0:
                correlation = (both_true / pred1_true).item()
                
                if correlation >= threshold:
                    discovered.append({
                        'premise': pred1.name,
                        'conclusion': pred2.name,
                        'confidence': correlation
                    })
    
    return discovered


def main():
    print("=" * 60)
    print("Tensor Logic - Rule Learning Example")
    print("=" * 60)
    print()
    
    # Create tensor logic engine
    tl = TensorLogic(domain_size=100)
    
    print("1. Creating training data...")
    
    # Define predicates
    has_fur = tl.predicate("has_fur", arity=1)
    has_whiskers = tl.predicate("has_whiskers", arity=1)
    purrs = tl.predicate("purrs", arity=1)
    is_cat = tl.predicate("is_cat", arity=1)
    is_mammal = tl.predicate("is_mammal", arity=1)
    
    # Training examples
    examples = [
        # Cats
        {"has_fur": 1.0, "has_whiskers": 1.0, "purrs": 1.0, "is_cat": 1.0, "is_mammal": 1.0},
        {"has_fur": 1.0, "has_whiskers": 1.0, "purrs": 0.9, "is_cat": 1.0, "is_mammal": 1.0},
        {"has_fur": 1.0, "has_whiskers": 1.0, "purrs": 1.0, "is_cat": 1.0, "is_mammal": 1.0},
        {"has_fur": 0.9, "has_whiskers": 1.0, "purrs": 0.8, "is_cat": 1.0, "is_mammal": 1.0},
        
        # Dogs (non-cats, but mammals)
        {"has_fur": 1.0, "has_whiskers": 0.7, "purrs": 0.0, "is_cat": 0.0, "is_mammal": 1.0},
        {"has_fur": 1.0, "has_whiskers": 0.6, "purrs": 0.0, "is_cat": 0.0, "is_mammal": 1.0},
        {"has_fur": 0.9, "has_whiskers": 0.5, "purrs": 0.0, "is_cat": 0.0, "is_mammal": 1.0},
        
        # Other animals
        {"has_fur": 0.0, "has_whiskers": 0.0, "purrs": 0.0, "is_cat": 0.0, "is_mammal": 0.0},
        {"has_fur": 0.0, "has_whiskers": 0.1, "purrs": 0.0, "is_cat": 0.0, "is_mammal": 0.0},
    ]
    
    # Add examples to knowledge base
    for idx, example in enumerate(examples):
        tl.fact(has_fur, idx, value=example["has_fur"])
        tl.fact(has_whiskers, idx, value=example["has_whiskers"])
        tl.fact(purrs, idx, value=example["purrs"])
        tl.fact(is_cat, idx, value=example["is_cat"])
        tl.fact(is_mammal, idx, value=example["is_mammal"])
    
    print(f"   Added {len(examples)} training examples")
    print()
    
    # Discover rules from data
    print("2. Discovering rules from data...")
    discovered = discover_rules(tl, threshold=0.7)
    
    print(f"   Found {len(discovered)} potential rules:")
    for rule in discovered:
        print(f"   - {rule['premise']} → {rule['conclusion']} (conf: {rule['confidence']:.2f})")
    
    print()
    
    # Add discovered rules to system
    print("3. Adding discovered rules to inference engine...")
    for rule in discovered:
        premise_pred = tl.kb.get_predicate(rule['premise'], arity=1)
        conclusion_pred = tl.kb.get_predicate(rule['conclusion'], arity=1)
        if premise_pred and conclusion_pred:
            tl.add_rule(premise_pred, conclusion_pred, confidence=rule['confidence'])
    
    print(f"   Added {len(discovered)} rules")
    print()
    
    # Test on new example
    print("4. Testing on new example...")
    # Ensure test index is within valid range
    test_idx = min(50, tl.kb.domain_size - 1)
    
    # New cat-like animal
    tl.fact(has_fur, test_idx, value=1.0)
    tl.fact(has_whiskers, test_idx, value=1.0)
    tl.fact(purrs, test_idx, value=0.9)
    
    print("   Input: has_fur=1.0, has_whiskers=1.0, purrs=0.9")
    print()
    
    # Run inference
    tl.infer()
    
    # Check predictions
    print("5. Predictions:")
    print("-" * 60)
    
    is_cat_pred = tl.query(is_cat, test_idx)
    is_mammal_pred = tl.query(is_mammal, test_idx)
    
    print(f"   is_cat: {is_cat_pred:.3f} {'✓' if is_cat_pred >= 0.7 else '✗'}")
    print(f"   is_mammal: {is_mammal_pred:.3f} {'✓' if is_mammal_pred >= 0.7 else '✗'}")
    
    print()
    print("=" * 60)
    print("Rule learning completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

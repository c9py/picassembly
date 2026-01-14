"""
Knowledge graph completion example.

Demonstrates using tensor logic for knowledge graph tasks like
link prediction and missing fact inference.
"""

import torch
from tensor_logic import TensorLogic


def main():
    print("=" * 60)
    print("Tensor Logic - Knowledge Graph Completion")
    print("=" * 60)
    print()
    
    # Create tensor logic engine for knowledge graph
    tl = TensorLogic(domain_size=200)
    
    print("1. Building knowledge graph...")
    
    # Define relation predicates
    parent_of = tl.predicate("parent_of", arity=2)
    sibling_of = tl.predicate("sibling_of", arity=2)
    grandparent_of = tl.predicate("grandparent_of", arity=2)
    ancestor_of = tl.predicate("ancestor_of", arity=2)
    
    print("   Relations: parent_of, sibling_of, grandparent_of, ancestor_of")
    print()
    
    # Add logical rules for knowledge graph reasoning
    print("2. Adding inference rules...")
    print("   - parent(X,Y) ∧ parent(Y,Z) → grandparent(X,Z)")
    print("   - parent(X,Z) → ancestor(X,Z)")
    print("   - grandparent(X,Z) → ancestor(X,Z)")
    print()
    
    tl.add_rule([parent_of, parent_of], grandparent_of, confidence=1.0)
    tl.add_rule(parent_of, ancestor_of, confidence=1.0)
    tl.add_rule(grandparent_of, ancestor_of, confidence=0.95)
    
    # Add known facts (family tree)
    print("3. Adding known facts...")
    
    family = {
        "parent_of": [
            ("alice", "bob", 1.0),
            ("alice", "charlie", 1.0),
            ("bob", "diana", 1.0),
            ("bob", "evan", 1.0),
            ("charlie", "frank", 1.0),
        ],
        "sibling_of": [
            ("bob", "charlie", 1.0),
            ("charlie", "bob", 1.0),
            ("diana", "evan", 1.0),
            ("evan", "diana", 1.0),
        ]
    }
    
    for rel, facts in family.items():
        pred = tl.kb.get_predicate(rel, arity=2)
        for subject, obj, conf in facts:
            tl.fact(pred, subject, obj, value=conf)
            print(f"   ✓ {rel}({subject}, {obj}) = {conf}")
    
    print()
    
    # Run inference to complete the knowledge graph
    print("4. Running inference to complete knowledge graph...")
    iterations = tl.infer(max_iterations=20)
    print(f"   Converged in {iterations} iterations")
    print()
    
    # Query inferred facts
    print("5. Inferred facts:")
    print("-" * 60)
    
    queries = [
        ("grandparent_of", "alice", "diana"),
        ("grandparent_of", "alice", "evan"),
        ("grandparent_of", "alice", "frank"),
        ("ancestor_of", "alice", "diana"),
        ("ancestor_of", "alice", "frank"),
        ("ancestor_of", "bob", "evan"),
    ]
    
    for rel, subj, obj in queries:
        pred = tl.kb.get_predicate(rel, arity=2)
        if pred:
            value = tl.query(pred, subj, obj)
            status = "✓" if value >= 0.5 else "✗"
            print(f"   {status} {rel}({subj}, {obj}) = {value:.2f}")
    
    print()
    print("=" * 60)
    print(f"Knowledge graph contains {tl.summary()['num_facts']} facts")
    print("=" * 60)


if __name__ == "__main__":
    main()

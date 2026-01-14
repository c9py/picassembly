"""
Comprehensive Tensor Logic Demo

This example demonstrates the full capabilities of the tensor logic framework,
combining multiple features in a single cohesive demonstration.
"""

import torch
from tensor_logic import TensorLogic


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  TENSOR LOGIC - Comprehensive Neural-Symbolic AI Demo".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Initialize
    print_section("1. INITIALIZATION")
    tl = TensorLogic(domain_size=200, device="cpu")
    print(f"✓ Created TensorLogic engine (domain_size=200)")
    print(f"✓ Using device: {tl.device}")
    
    # Define predicates
    print_section("2. PREDICATE DEFINITION")
    
    # Biological taxonomy
    animal = tl.predicate("animal", arity=1)
    mammal = tl.predicate("mammal", arity=1)
    cat = tl.predicate("cat", arity=1)
    dog = tl.predicate("dog", arity=1)
    
    # Properties
    has_fur = tl.predicate("has_fur", arity=1)
    warm_blooded = tl.predicate("warm_blooded", arity=1)
    can_swim = tl.predicate("can_swim", arity=1)
    carnivore = tl.predicate("carnivore", arity=1)
    
    print("✓ Defined 8 predicates:")
    print("  - Taxonomy: animal, mammal, cat, dog")
    print("  - Properties: has_fur, warm_blooded, can_swim, carnivore")
    
    # Add taxonomic rules
    print_section("3. LOGICAL RULES (Taxonomy & Properties)")
    
    rules = [
        (mammal, animal, 1.0, "mammal(X) → animal(X)"),
        (cat, mammal, 1.0, "cat(X) → mammal(X)"),
        (dog, mammal, 1.0, "dog(X) → mammal(X)"),
        (mammal, warm_blooded, 1.0, "mammal(X) → warm_blooded(X)"),
        (mammal, has_fur, 0.95, "mammal(X) → has_fur(X) [95%]"),
        (cat, carnivore, 0.9, "cat(X) → carnivore(X) [90%]"),
        (dog, carnivore, 0.7, "dog(X) → carnivore(X) [70%]"),
    ]
    
    for premise, conclusion, conf, desc in rules:
        tl.add_rule(premise, conclusion, confidence=conf)
        print(f"✓ {desc}")
    
    # Add facts
    print_section("4. ASSERTING FACTS")
    
    facts = [
        (cat, "whiskers", 1.0),
        (cat, "mittens", 0.95),
        (dog, "rex", 1.0),
        (dog, "buddy", 0.98),
        (animal, "goldfish", 1.0),  # Fish, not a mammal
    ]
    
    for pred, entity, value in facts:
        tl.fact(pred, entity, value=value)
        print(f"✓ {pred.name}({entity}) = {value:.2f}")
    
    # Inference
    print_section("5. RUNNING INFERENCE")
    print("Starting forward chaining inference...")
    
    iterations = tl.infer(max_iterations=20)
    print(f"✓ Converged in {iterations} iterations")
    
    # Query results
    print_section("6. QUERY RESULTS")
    
    test_cases = [
        ("whiskers", [animal, mammal, cat, warm_blooded, has_fur, carnivore]),
        ("rex", [animal, mammal, dog, warm_blooded, has_fur, carnivore]),
        ("goldfish", [animal, mammal, warm_blooded]),
    ]
    
    for entity, predicates_to_check in test_cases:
        print(f"\n{entity.upper()}:")
        print("-" * 70)
        for pred in predicates_to_check:
            value = tl.query(pred, entity)
            status = "✓" if value >= 0.5 else "✗"
            confidence = "HIGH" if value >= 0.9 else "MED" if value >= 0.5 else "LOW"
            print(f"  {status} {pred.name:15s} = {value:.3f} [{confidence}]")
    
    # Fuzzy logic demonstration
    print_section("7. FUZZY LOGIC DEMONSTRATION")
    
    test_animal = tl.predicate("test_animal", arity=1)
    tl.fact(test_animal, "unknown", value=0.7)  # 70% confidence it's an animal
    
    # Add fuzzy rule
    tl.add_rule(test_animal, mammal, confidence=0.8)
    tl.infer()
    
    result = tl.query(mammal, "unknown")
    print(f"If 70% sure it's an animal, and animals → mammals with 80% confidence")
    print(f"Then we're {result:.1%} sure it's a mammal")
    print(f"(Fuzzy logic: 0.7 * 0.8 = {0.7*0.8:.2f})")
    
    # Statistics
    print_section("8. SYSTEM STATISTICS")
    
    summary = tl.summary()
    print(f"Predicates:     {summary['num_predicates']}")
    print(f"Rules:          {summary['num_rules']}")
    print(f"Facts:          {summary['num_facts']}")
    print(f"Device:         {summary['device']}")
    
    # Demonstrate neural integration
    print_section("9. NEURAL-SYMBOLIC INTEGRATION")
    
    from tensor_logic.neural.modules import NeuralLogicModule
    
    model = NeuralLogicModule(
        input_dim=20,
        output_dim=3,
        hidden_dims=[16, 8],
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Created neural-logic module")
    print(f"  - Input dim: 20")
    print(f"  - Output dim: 3")
    print(f"  - Parameters: {num_params:,}")
    
    # Test forward pass
    test_input = torch.randn(5, 20)
    output = model(test_input)
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {tuple(test_input.shape)}")
    print(f"  - Output shape: {tuple(output.shape)}")
    print(f"  - Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Final summary
    print_section("10. CONCLUSION")
    
    print("\n✓ Tensor Logic Framework Capabilities Demonstrated:")
    print("  [✓] Symbolic reasoning with predicates and rules")
    print("  [✓] Fuzzy logic with continuous truth values")
    print("  [✓] Differentiable operations for neural integration")
    print("  [✓] Forward chaining inference")
    print("  [✓] Knowledge representation and querying")
    print("  [✓] Neural-symbolic module integration")
    print("  [✓] Scalable tensor operations")
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  Demo Complete - Tensor Logic is Ready for AGI!".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()

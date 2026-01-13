"""
Basic reasoning example with Tensor Logic.

Demonstrates simple logical inference with predicates and rules.
"""

from tensor_logic import TensorLogic


def main():
    print("=" * 60)
    print("Tensor Logic - Basic Reasoning Example")
    print("=" * 60)
    print()
    
    # Create tensor logic engine
    tl = TensorLogic(domain_size=100)
    
    # Define predicates
    print("1. Defining predicates...")
    human = tl.predicate("human", arity=1)
    mortal = tl.predicate("mortal", arity=1)
    philosopher = tl.predicate("philosopher", arity=1)
    wise = tl.predicate("wise", arity=1)
    
    # Add rules
    print("2. Adding logical rules...")
    print("   - human(X) → mortal(X)")
    print("   - philosopher(X) → human(X)")
    print("   - philosopher(X) → wise(X)")
    print()
    
    tl.add_rule(human, mortal, confidence=1.0)
    tl.add_rule(philosopher, human, confidence=1.0)
    tl.add_rule(philosopher, wise, confidence=0.9)
    
    # Add facts
    print("3. Asserting facts...")
    print("   - philosopher(socrates) = 1.0")
    print("   - philosopher(plato) = 1.0")
    print("   - philosopher(aristotle) = 1.0")
    print()
    
    tl.fact(philosopher, "socrates", value=1.0)
    tl.fact(philosopher, "plato", value=1.0)
    tl.fact(philosopher, "aristotle", value=1.0)
    
    # Run inference
    print("4. Running inference...")
    iterations = tl.infer(max_iterations=10)
    print(f"   Converged in {iterations} iterations")
    print()
    
    # Query results
    print("5. Query results:")
    print("-" * 60)
    
    philosophers = ["socrates", "plato", "aristotle"]
    predicates = [("human", human), ("mortal", mortal), ("wise", wise)]
    
    for person in philosophers:
        print(f"\n{person.capitalize()}:")
        for pred_name, pred in predicates:
            value = tl.query(pred, person)
            status = "✓" if value >= 0.5 else "✗"
            print(f"  {status} {pred_name}({person}) = {value:.2f}")
    
    print()
    print("=" * 60)
    print("Summary:")
    summary = tl.summary()
    print(f"  Predicates: {summary['num_predicates']}")
    print(f"  Rules: {summary['num_rules']}")
    print(f"  Facts: {summary['num_facts']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

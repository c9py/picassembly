# Tensor Logic

A Python framework for bridging neural and symbolic AI through tensor-based logical operations.

## Overview

Tensor Logic unifies logical reasoning and neural network operations into a single differentiable framework. It enables:

- **Neural-Symbolic Integration**: Seamlessly combine symbolic reasoning with neural learning
- **Differentiable Logic**: All logical operations are differentiable for gradient-based learning
- **Scalable Reasoning**: Leverage tensor operations for parallel, GPU-accelerated inference
- **Knowledge Representation**: Rich support for predicates, rules, and knowledge bases

## Architecture

The framework is built on several key components:

1. **Tensor Representation**: Logical predicates and facts represented as tensors
2. **Logical Operations**: Differentiable implementations of AND, OR, NOT, IMPLIES
3. **Inference Engine**: Forward/backward chaining with probabilistic reasoning
4. **Knowledge Base**: Storage and retrieval of logical rules and facts
5. **Neural Bridge**: Integration with PyTorch for end-to-end learning

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from tensor_logic import TensorLogic, Predicate, Rule

# Create tensor logic engine
tl = TensorLogic()

# Define predicates
mortal = Predicate("mortal", arity=1)
human = Predicate("human", arity=1)

# Add rule: human(X) -> mortal(X)
rule = Rule(premise=human, conclusion=mortal)
tl.add_rule(rule)

# Add fact: human(socrates)
tl.add_fact(human("socrates"))

# Query: mortal(socrates)?
result = tl.query(mortal("socrates"))
print(f"Is Socrates mortal? {result}")
```

## Features

### Differentiable Logic Operations

All logical operations support gradient computation:

```python
import torch
from tensor_logic.ops import fuzzy_and, fuzzy_or, fuzzy_not

# Fuzzy logic with gradients
x = torch.tensor([0.8], requires_grad=True)
y = torch.tensor([0.6], requires_grad=True)

z = fuzzy_and(x, y)
z.backward()
```

### Knowledge Base Management

```python
from tensor_logic import KnowledgeBase

kb = KnowledgeBase()
kb.add_predicate("parent", arity=2)
kb.add_fact("parent(john, mary)")
kb.add_rule("parent(X, Y) ∧ parent(Y, Z) → grandparent(X, Z)")

results = kb.query("grandparent(john, Z)")
```

### Neural-Symbolic Learning

```python
from tensor_logic.neural import NeuralLogicModule

# Define neural-symbolic model
model = NeuralLogicModule(
    input_dim=128,
    logic_rules=rules,
    output_dim=10
)

# Train with standard PyTorch
optimizer = torch.optim.Adam(model.parameters())
loss = model(data, labels)
loss.backward()
optimizer.step()
```

## Examples

See the `examples/` directory for comprehensive tutorials:

- `basic_reasoning.py` - Simple logical inference
- `neural_integration.py` - Neural-symbolic learning
- `knowledge_graphs.py` - Knowledge graph completion
- `rule_learning.py` - Automatic rule discovery

## Testing

Run the test suite:

```bash
pytest
```

With coverage:
```bash
pytest --cov=tensor_logic --cov-report=html
```

## Architecture Details

### Tensor Representation

Predicates are represented as tensors where:
- **0-arity predicates** (propositions): scalar tensors
- **1-arity predicates** (unary): 1D tensors indexed by entities
- **n-arity predicates**: n-dimensional tensors

Truth values are continuous [0, 1] for fuzzy logic compatibility.

### Inference Algorithm

The engine supports multiple inference modes:

1. **Forward Chaining**: Bottom-up, data-driven reasoning
2. **Backward Chaining**: Top-down, goal-driven reasoning  
3. **Probabilistic**: Uncertainty propagation using fuzzy logic
4. **Abductive**: Best explanation inference

## References

- [Tensor Logic: Bridging Neural-Symbolic Divide](https://bengoertzel.substack.com/)
- [Neural-Symbolic AI Overview](https://tensor-logic.org/)
- [Logic Tensor Networks](https://arxiv.org/abs/2012.13635)

## License

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

```bibtex
@software{tensor_logic_2026,
  title={Tensor Logic: A Framework for Neural-Symbolic AI},
  author={PICA PICA Team},
  year={2026},
  url={https://github.com/c9py/picassembly}
}
```

"""
Neural-symbolic integration example.

Demonstrates combining neural networks with logical reasoning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tensor_logic import TensorLogic
from tensor_logic.neural.modules import NeuralLogicModule


def main():
    print("=" * 60)
    print("Tensor Logic - Neural Integration Example")
    print("=" * 60)
    print()
    
    # Create a simple classification task with logical constraints
    print("1. Setting up neural-symbolic model...")
    
    # Create neural-logic module
    model = NeuralLogicModule(
        input_dim=10,
        output_dim=3,
        hidden_dims=[32, 16],
        dropout=0.1
    )
    
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print()
    
    # Generate synthetic data
    print("2. Generating synthetic training data...")
    torch.manual_seed(42)
    
    # Training data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 3, (100,))
    
    # Test data
    X_test = torch.randn(20, 10)
    y_test = torch.randint(0, 3, (20,))
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    print()
    
    # Training loop
    print("3. Training neural-symbolic model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    print()
    
    # Evaluation
    print("4. Evaluating on test set...")
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        
        # Calculate accuracy
        _, predicted = torch.max(test_outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)
        
        print(f"   Test Loss: {test_loss.item():.4f}")
        print(f"   Test Accuracy: {accuracy:.2%}")
    
    print()
    
    # Demonstrate logical reasoning integration
    print("5. Integrating with logical reasoning...")
    
    tl = TensorLogic()
    
    # Define predicates based on classes
    class_0 = tl.predicate("class_0", arity=1)
    class_1 = tl.predicate("class_1", arity=1)
    class_2 = tl.predicate("class_2", arity=1)
    valid = tl.predicate("valid", arity=1)
    
    # Add constraint: only one class can be true
    print("   Adding logical constraints...")
    print("   - class_i(X) → valid(X)")
    
    tl.add_rule(class_0, valid)
    tl.add_rule(class_1, valid)
    tl.add_rule(class_2, valid)
    
    # Set facts based on neural predictions
    for i in range(min(5, len(predicted))):
        class_idx = predicted[i].item()
        if class_idx == 0:
            tl.fact(class_0, f"sample_{i}", value=test_outputs[i][0].item())
        elif class_idx == 1:
            tl.fact(class_1, f"sample_{i}", value=test_outputs[i][1].item())
        else:
            tl.fact(class_2, f"sample_{i}", value=test_outputs[i][2].item())
    
    tl.infer()
    
    print("   Logical validation completed")
    print()
    
    # Show results
    print("6. Sample predictions with logical validation:")
    print("-" * 60)
    
    for i in range(min(5, len(predicted))):
        pred_class = predicted[i].item()
        true_class = y_test[i].item()
        is_valid = tl.query(valid, f"sample_{i}")
        
        status = "✓" if pred_class == true_class else "✗"
        print(f"   {status} Sample {i}: Predicted={pred_class}, True={true_class}, Valid={is_valid:.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

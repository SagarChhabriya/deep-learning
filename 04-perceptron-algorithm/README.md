# **4. Perceptron Learning Algorithm**

## **4.1 The Perceptron Trick**
A geometric approach to iteratively adjust the decision boundary for misclassified points.

### **Key Mechanism:**
1. For a misclassified point **x** with label **y**:
   - If predicted ŷ=0 but y=1:  
     **New weights** = Old weights + x  
     (Moves boundary *toward* the point)
   - If predicted ŷ=1 but y=0:  
     **New weights** = Old weights - x  
     (Moves boundary *away* from the point)

2. Mathematical Form:  
   $$
   w_{new} = w_{old} + \alpha \cdot (y - \hat{y}) \cdot x
   $$
   Where α = learning rate (typically 1 for standard perceptron)

## **4.2 Training Process**
1. Initialize weights (w) randomly or to zero
2. For each training epoch:
   - Classify all points using current weights
   - Update weights for misclassified points
3. Repeat until convergence (or max epochs)

## **4.3 Model Versatility**
The perceptron is a foundational mathematical model capable of solving various machine learning problems, with its functionality determined by the activation function:
- **Binary Classification**: Step activation (original perceptron)
- **Probabilistic Classification**: Sigmoid activation (logistic regression)
- **Multiclass Problems**: Softmax activation
- **Regression Tasks**: Linear activation

This flexibility makes it adaptable to:
- Simple linear decision boundaries
- Multiple feature scenarios (through vectorized implementations)
- Both separable and non-separable problems (with appropriate extensions)

## **4.4 Flexibility Considerations**
- **Learning Rate (α):** Controls step size of boundary adjustment
- **Feature Scaling:** Normalization improves convergence
- **Maximum Epochs:** Prevents infinite loops for non-separable data

## **4.5 Comparison of Common Classifiers**

| Loss Function          | Activation | Model Type               | Output Type               |
|------------------------|------------|--------------------------|---------------------------|
| Hinge Loss             | Step       | Perceptron               | Binary Classification (0/1) |
| Log Loss (Binary CE)   | Sigmoid    | Logistic Regression      | Binary Probability [0,1]  |
| Categorical CE         | Softmax    | Softmax Regression       | Multiclass Probability    |
| Mean Squared Error     | Linear     | Linear Regression        | Continuous Value          |

### **Key Insights:**
1. **Perceptron** uses hinge loss with hard decisions
2. **Logistic Regression** provides probabilistic outputs
3. **Softmax** extends to multiple classes
4. **Linear Regression** solves continuous problems

## **4.6 Practical Implications**
- The perceptron trick forms the basis for modern gradient descent
- Demonstrates how simple weight updates can learn patterns
- Highlights importance of proper loss-activation pairing
- Serves as the prototype for more complex neural architectures

### **Visualization:**
![](assets/perceptron-trick.gif)

# **7. Backpropagation Algorithm**

## **7.1 Introduction**
Backpropagation (backward propagation of errors) is a fundamental algorithm for training artificial neural networks through supervised learning. It efficiently computes the gradient of the loss function with respect to each weight in the network using the chain rule of calculus, enabling optimization via gradient descent.

### Key Characteristics:
- **Supervised Learning**: Requires labeled training data
- **Gradient-Based**: Computes error derivatives
- **Recursive**: Applies chain rule layer-by-layer
- **Efficient**: Avoids redundant calculations

## **7.2 Mathematical Foundations**

### **Core Components:**
1. **Forward Pass:**
   - Compute network outputs from inputs to outputs
   - Store intermediate activation values

2. **Loss Calculation:**
   - Compare predictions (ŷ) with true values (y)
   - Compute error using loss function (e.g., MSE, Cross-Entropy)

3. **Backward Pass:**
   - Propagate error gradients backward through the network
   - Compute weight updates using chain rule

### **Mathematical Formulation:**

#### **Notation:**
- $w^l_{jk}$: Weight from neuron k in layer (l-1) to neuron j in layer l
- $z^l_j$: Weighted input to neuron j in layer l
- $a^l_j$: Activation of neuron j in layer l ($a^l_j = \sigma(z^l_j)$)
- $\delta^l_j$: Error at neuron j in layer l

#### **Key Equations:**
1. **Output Layer Error (L):**

$$
   \delta^L_j = \frac{\partial C}{\partial a^L_j} \sigma'(z^L_j)
$$

   Where C is the cost function

2. **Hidden Layer Error (l):**

$$
   \delta^l_j = \sum_k w^{l+1}_{kj} \delta^{l+1}_k \sigma'(z^l_j)
$$

3. **Weight Gradient:**

$$
   \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
$$

4. **Bias Gradient:**

$$
   \frac{\partial C}{\partial b^l_j} = \delta^l_j
$$

### **The Chain Rule in Action:**
For a simple 2-layer network with MSE loss:

$$
\frac{\partial C}{\partial w} = \underbrace{\frac{\partial C}{\partial a^L}}_{\text{Loss gradient}} \times \underbrace{\frac{\partial a^L}{\partial z^L}}_{\text{Activation derivative}} \times \underbrace{\frac{\partial z^L}{\partial w}}_{\text{Input from prev. layer}}
$$

## **7.3 Algorithm Steps**

1. **Initialization:**
   - Set all weights to small random values
   - Choose learning rate η

2. **Training Loop:**
   ```
   for each training example (x,y):
       # Forward pass
       a¹ = x
       for l = 2 to L:
           zˡ = Wˡaˡ⁻¹ + bˡ
           aˡ = σ(zˡ)
       
       # Backward pass
       δᴸ = ∇ₐC ⊙ σ'(zᴸ)
       for l = L-1 to 2:
           δˡ = (Wˡ⁺¹ᵀδˡ⁺¹) ⊙ σ'(zˡ)
       
       # Weight updates
       for l = 2 to L:
           Wˡ -= η(δˡaˡ⁻¹ᵀ)
           bˡ -= ηδˡ
   ```

## **7.4 Practical Considerations**

### **Computational Efficiency:**
- **Matrix Operations**: Implemented as batch operations
- **Activation Derivatives**: Common choices:
  - Sigmoid: σ'(z) = σ(z)(1-σ(z))
  - ReLU: σ'(z) = 1 if z > 0 else 0
  - Softmax: Special case for output layer

### **Common Challenges:**
1. **Vanishing Gradients**: 
   - Occurs with deep networks and certain activations (e.g., sigmoid)
   - Mitigation: Use ReLU, residual connections

2. **Exploding Gradients**:
   - Large weight values cause numerical instability
   - Mitigation: Gradient clipping, weight regularization

3. **Local Minima**:
   - Network may converge to suboptimal solutions
   - Mitigation: Momentum, adaptive learning rates

## **7.5 Modern Variations**

| **Variant**         | **Key Feature**                     | **Advantage**                      |
|---------------------|-----------------------------------|-----------------------------------|
| Stochastic GD       | Updates per sample                | Faster convergence                |
| Mini-batch GD       | Updates per small batch           | Balance speed/stability           |
| Momentum            | Accumulates past gradients        | Avoids local minima               |
| RMSprop             | Adaptive learning rates           | Handles sparse gradients          |
| Adam                | Combines momentum + RMSprop       | Most widely used optimizer        |

## **7.6 Implementation Notes**
- All modern deep learning frameworks (TensorFlow, PyTorch) provide automatic differentiation
- The algorithm generalizes to any differentiable architecture (CNNs, RNNs)
- Computational graphs track operations for efficient gradient computation

https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll

 
# **3. Perceptron**

## **3.1 Introduction to Perceptron**
- **Definition**:  
  A perceptron is a supervised learning algorithm for binary classification, forming the foundational unit of neural networks. Inspired by biological neurons, it processes inputs to produce binary outputs (0 or 1).

- **Properties**:  
  - Designed for **linearly separable data**.  
  - Serves as the building block for multilayer architectures in deep learning.  
  - Functionally mimics a biological neuron:  
    - **Inputs** (dendrites) → **Weights** (synaptic strength) → **Activation** (cell body) → **Output** (axon).  

- **Geometric Intuition**:  
  The perceptron learns a **hyperplane** (e.g., a line in 2D) to separate data points into two classes. For input features $x_1, x_2$, the decision boundary is $w_1x_1 + w_2x_2 + b = 0$, where $w$ are weights and $b$ is the bias.  
  ![](assets/01-perceptron.png)

---

## **3.2 Training the Perceptron**
1. **Initialization**:  
   - Weights $w$ and bias $b$ are initialized randomly or to zero.  

2. **Forward Pass**:  
   - Compute the weighted sum: $z = \sum (w_i x_i) + b$.  
   - Apply the **step function** (activation):  

$$
     \text{Output} = 
     \begin{cases} 
     1 & \text{if } z \geq 0, \\
     0 & \text{otherwise}.
     \end{cases}
$$

3. **Weight Update Rule**:  
   - For misclassified samples, adjust weights and bias:  

$$
     w_i = w_i + \alpha \cdot (y - \hat{y}) \cdot x_i, \quad b = b + \alpha \cdot (y - \hat{y})
$$  

     where $\alpha$ = learning rate, $y$ = true label, $\hat{y}$ = predicted label.  

4. **Convergence**:  
   - Guaranteed only if data is **linearly separable** (Rosenblatt, 1958).  

---

## **3.3 Limitations and Solutions**
### **Problems with Perceptron**:
1. **Linear Separability Constraint**:  
   Fails on non-linear data (e.g., XOR problem).  
2. **Binary Output**:  
   Cannot handle probabilistic predictions.  
3. **No Hidden Layers**:  
   Limited to single-layer decision boundaries.  
   

### **Solutions: Multilayer Perceptron (MLP)**:
1. **Architecture**:  
   - Adds **hidden layers** between input and output.  
   - Uses **non-linear activation functions** (e.g., ReLU, Sigmoid).  
2. **Capabilities**:  
   - Solves XOR and other non-linear problems.  
   - Enables feature hierarchy learning (shallow → abstract features).  
3. **Training**:  
   - Employs **backpropagation** for weight optimization.  

---

### **Key Takeaways**
- The perceptron is a **linear classifier** with historical significance in neural networks.  
- Its inability to handle non-linearity led to the development of **MLPs** and modern deep learning.  
- MLPs overcome perceptron limitations through **stacked layers** and **non-linear transformations**.  

---

### **References**
1. Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization*.  
2. Minsky, M. & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*.  
3. Nielsen, M. (2015). *Neural Networks and Deep Learning*, Chapter 1. [Online Book](http://neuralnetworksanddeeplearning.com/chap1.html). 
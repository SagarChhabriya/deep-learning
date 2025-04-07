# **8. Memoization in Deep Learning**

## **8.1 Introduction to Memoization**
Memoization is an optimization technique that stores computationally expensive function call results and returns the cached result when the same inputs occur again. In deep learning, this principle is crucial for:

- **Efficient backpropagation** (storing intermediate values during forward pass)
- **Recurrent network computations** (avoiding redundant calculations)
- **Hyperparameter optimization** (caching model evaluations)

## **8.2 Mathematical Foundation**
For a function *f: X → Y*, memoization creates a mapping:

$$
M: X \rightarrow Y \quad \text{where} \quad M(x) = 
\begin{cases} 
f(x) & \text{if } x \notin \text{keys}(M) \\
M[x] & \text{otherwise}
\end{cases}
$$

## **8.3 Key Applications in Deep Learning**

### **1. Backpropagation Efficiency**
During forward pass:
- Store all layer activations $(a^1, a^2, ..., a^L)$
- Cache weighted inputs $(z^1, z^2, ..., z^L)$
- Avoids recomputation during backward pass

**Compute Savings**:
- Without memoization: $O(2L)$ computations
- With memoization: $O(L)$ computations

### **2. Recursive Network Architectures**
For RNNs processing sequence $(x_1, ..., x_T)$:
- Memoize hidden states $h_t$ at each timestep
- Enables efficient truncated backpropagation through time (TBTT)

### **3. Attention Mechanisms**
In Transformer architectures:
- Key-Value pairs are memoized
- Allows constant-time lookup during self-attention

## **8.4 Implementation Techniques**

| **Technique**       | **Application**                  | **DL Framework Example**         |
|---------------------|----------------------------------|----------------------------------|
| Activation caching  | Backpropagation                  | `tensor.retain_grad()` (PyTorch) |
| Checkpointing       | Memory optimization              | `torch.utils.checkpoint`         |
| Hash-based caching  | Hyperparameter search            | `functools.lru_cache`            |
| Gradient tapes      | Automatic differentiation       | `tf.GradientTape` (TensorFlow)   |

## **8.5 Memory-Speed Tradeoffs**

**Pros**:
- Reduces computation time by 30-50% for many architectures
- Enables training of deeper networks
- Critical for real-time inference systems

**Cons**:
- Increases memory usage (stores intermediate values)
- Requires careful memory management in resource-constrained environments
- Can complicate distributed training implementations

## **8.6 Advanced Memoization Strategies**

1. **Selective Memoization**:
   - Only cache layers with high computational cost
   - Example: Cache transformer attention layers but not embeddings

2. **Approximate Memoization**:
   - Store quantized/low-precision versions of activations
   - Used in mobile/edge deployments

3. **Dynamic Programming**:
   - Memoization forms basis for many DP algorithms
   - Applied in neural architecture search (NAS)

## **8.7 Case Study: Transformer Inference**
```python
class MemoizedAttention(nn.Module):
    def __init__(self):
        self.kv_cache = None  # Memoization store
        
    def forward(self, x):
        if self.kv_cache is None:
            # Compute and store key-value pairs
            self.kv_cache = self.compute_kv(x)
        return self.attention(x, self.kv_cache)
```
**Impact**:
- Reduces inference computation from $O(n^2)$ to $O(1)$ for cached tokens
- Enables efficient autoregressive generation



## **8.8 Fibonacci Code Example**

```py
import time
from functools import lru_cache

# 1. Fibonacci without memoization (inefficient recursive approach)
def fib_no_memo(n):
    if n <= 1:
        return n
    return fib_no_memo(n-1) + fib_no_memo(n-2)

# 2. Fibonacci with manual memoization
def fib_manual_memo(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_manual_memo(n-1, memo) + fib_manual_memo(n-2, memo)
    return memo[n]

# 3. Fibonacci with built-in memoization decorator
@lru_cache(maxsize=None)
def fib_lru_cache(n):
    if n <= 1:
        return n
    return fib_lru_cache(n-1) + fib_lru_cache(n-2)

# Test function to compare performance
def test_fib(fib_func, n, name):
    start_time = time.time()
    result = fib_func(n)
    end_time = time.time()
    print(f"{name}:")
    print(f"Fib({n}) = {result}")
    print(f"Execution time: {end_time - start_time:.6f} seconds\n")

# Compare all approaches
n = 35  # Large enough to show significant time difference

print(f"Calculating Fibonacci({n}) with different approaches:\n")
test_fib(fib_no_memo, n, "Standard recursive (no memoization)")
test_fib(fib_manual_memo, n, "Manual memoization")
test_fib(fib_lru_cache, n, "LRU cache memoization")

# Clear cache for fair comparison between runs
fib_lru_cache.cache_clear()
```

```js
Calculating Fibonacci(35) with different approaches:

Standard recursive (no memoization):
Fib(35) = 9227465
Execution time: 3.742315 seconds

Manual memoization:
Fib(35) = 9227465
Execution time: 0.000047 seconds

LRU cache memoization:
Fib(35) = 9227465
Execution time: 0.000029 seconds
```



## **8.9 Future Directions**
- **Differentiable memoization**: Learning what to cache
- **Hardware-optimized caching**: Using GPU shared memory
- **Federated caching**: For distributed training scenarios

Memoization remains a fundamental technique that enables modern deep learning systems to achieve both computational efficiency and practical scalability.
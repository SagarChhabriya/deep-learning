## 🧪 Dummy Dataset (Binary Classification)


| x1 | x2 | y |
| -- | -- | - |
| 0  | 0  | 0 |
| 0  | 1  | 0 |
| 1  | 0  | 0 |
| 1  | 1  | 1 |
| 2  | 2  | 1 |

### 1. **Model Initialization**

```python
p = Perceptron(eta=0.1, n_iter=2, random_state=1)
```

* Learning rate: 0.1
* Epochs: 2 (for simplicity)
* Random seed: 1

---


### Data

```python
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1],
              [2, 2]])

y = np.array([0, 0, 0, 1, 1])
```

---

### 🎲 Initial Weights

From:

```python
self.w_ = rgen.normal(loc=0.0, scale=0.01, size=2)
self.b_ = 0.0
```

Using `random_state=1`, the generated weights are approximately:

```python
w_ = [ 0.01624345, -0.00611756 ]
b_ = 0.0
```

---

## 🔁 Epoch 1

### Step through each sample:

#### 🟡 Sample 1: `[0, 0]`, Target = 0

* Net input: `0*0.0162 + 0*(-0.0061) + 0 = 0`
* Prediction: `1` (since 0 ≥ 0 → step function)
* Error: `0 - 1 = -1`
* **Update**: `-0.1`

  * `w_ = w_ + (-0.1) * [0, 0] = no change`
  * `b_ = b_ + (-0.1) = -0.1`
* **Errors**: 1

---

#### 🟡 Sample 2: `[0, 1]`, Target = 0

* Net input: `0*0.0162 + 1*(-0.0061) - 0.1 = -0.1061`
* Prediction: `0`
* Target = Prediction → no update
* **Errors**: 1

---

#### 🟡 Sample 3: `[1, 0]`, Target = 0

* Net input: `1*0.0162 + 0 - 0.1 = -0.0838`
* Prediction: `0`
* Target = Prediction → no update
* **Errors**: 1

---

#### 🟡 Sample 4: `[1, 1]`, Target = 1

* Net input: `0.0162 + (-0.0061) - 0.1 = -0.0899`
* Prediction: `0`
* Error: `1 - 0 = +1`
* **Update**: `+0.1`

  * `w_ = w_ + 0.1 * [1, 1] → [0.1162, 0.0939]`
  * `b_ = -0.1 + 0.1 = 0.0`
* **Errors**: 2

---

#### 🟡 Sample 5: `[2, 2]`, Target = 1

* Net input: `2*0.1162 + 2*0.0939 + 0 = 0.4202`
* Prediction: `1`
* No update
* **Errors**: 2

---

### ✅ After Epoch 1:

* `w_ = [0.1162, 0.0939]`
* `b_ = 0.0`
* `errors_ = [2]`

---

## 🔁 Epoch 2

#### 🟢 Sample 1: `[0, 0]`, Target = 0

* Net input = 0
* Prediction: `1`
* Error: -1
* Update: -0.1

  * `b_ = -0.1`
* **Errors**: 1

---

#### 🟢 Sample 2: `[0, 1]`, Target = 0

* Net input = `0.0939 - 0.1 = -0.0061`
* Prediction: 0 → correct
* No update
* **Errors**: 1

---

#### 🟢 Sample 3: `[1, 0]`, Target = 0

* Net input = `0.1162 - 0.1 = 0.0162`
* Prediction: 1 → incorrect
* Update: -0.1

  * `w_[0] = 0.0162`
  * `b_ = -0.2`
* **Errors**: 2

---

#### 🟢 Sample 4: `[1, 1]`, Target = 1

* Net input = `0.0162 + 0.0939 - 0.2 = -0.0899`
* Prediction: 0 → wrong
* Update: +0.1

  * `w_ = [0.1162, 0.1939]`
  * `b_ = -0.1`
* **Errors**: 3

---

#### 🟢 Sample 5: `[2, 2]`, Target = 1

* Net input = `2*0.1162 + 2*0.1939 - 0.1 ≈ 0.6202`
* Prediction: 1 → correct
* No update
* **Errors**: 3

---

## ✅ Final State After 2 Epochs

* Final weights: `[0.1162, 0.1939]`
* Final bias: `-0.1`
* Errors per epoch: `[2, 3]`

---

## 🧾 Summary

* The model **started with random weights**.
* It **updated weights only when predictions were wrong**.
* The weight vector adjusted **toward making correct predictions**.
* After 2 epochs, the model was already making mostly correct predictions.

---
### Explanation
Great question! Let's break down the **error calculation** and **weight update** process in detail. We’ll go through both the **epoch-wise error count** and the **weight update mechanism**.

---

### 🔍 **Error Calculation**

The **error** represents the difference between the **predicted output** and the **true target**.

**Formula for error**:

$$
\text{error} = \text{target} - \text{prediction}
$$

---

### Step-by-Step: **Epoch 1 (Error Calculation)**

#### **Sample 1**:

* **Input**: `[0, 0]`, Target: `0`
* **Prediction**: `1` (since the net input is `0`, the step function outputs `1`).
* **Error**: `0 - 1 = -1`
* **Update**: Bias `b` is updated by `-0.1`, but weights don’t change (since the input values are `[0, 0]`).
* **Error Count after Sample 1**: 1 error.

#### **Sample 2**:

* **Input**: `[0, 1]`, Target: `0`
* **Prediction**: `0` (since net input is negative).
* **Error**: `0 - 0 = 0` (no error).
* **Update**: No update needed, because the prediction matches the target.
* **Error Count after Sample 2**: 1 error.

#### **Sample 3**:

* **Input**: `[1, 0]`, Target: `0`
* **Prediction**: `0` (since net input is negative).
* **Error**: `0 - 0 = 0` (no error).
* **Update**: No update needed.
* **Error Count after Sample 3**: 1 error.

#### **Sample 4**:

* **Input**: `[1, 1]`, Target: `1`
* **Prediction**: `0` (since net input is negative).
* **Error**: `1 - 0 = +1` (there's an error).
* **Update**: Weights and bias are updated (weights increase).
* **Error Count after Sample 4**: 2 errors.

#### **Sample 5**:

* **Input**: `[2, 2]`, Target: `1`
* **Prediction**: `1` (net input is positive).
* **Error**: `1 - 1 = 0` (no error).
* **Update**: No update needed.
* **Error Count after Sample 5**: 2 errors.

#### **Epoch 1 Summary**:

* The total number of errors after **Epoch 1** is `[2]` (which means 2 samples were incorrectly classified during the epoch).

---

### Step-by-Step: **Epoch 2 (Error Calculation)**

#### **Sample 1** (Epoch 2):

* **Input**: `[0, 0]`, Target: `0`
* **Prediction**: `1` (since `net_input = 0`).
* **Error**: `0 - 1 = -1` (there is an error).
* **Update**: Bias is updated by `-0.1`:

$$
b = b + \eta \cdot \text{error} = 0.0 - 0.1 = -0.1
$$

* **Error Count after Sample 1**: 1 error.

#### **Sample 2** (Epoch 2):

* **Input**: `[0, 1]`, Target: `0`
* **Prediction**: `0` (since net input is negative).
* **Error**: `0 - 0 = 0` (no error).
* **Update**: No update needed.
* **Error Count after Sample 2**: 1 error.

#### **Sample 3** (Epoch 2):

* **Input**: `[1, 0]`, Target: `0`
* **Prediction**: `1` (since `net_input = 0.0162`).
* **Error**: `0 - 1 = -1` (there is an error).
* **Update**: Update weights and bias.

  * **Weight update**:

$$
w_0 = w_0 + \eta \cdot \text{error} \cdot x_0 = 0.0162 + 0.1 \cdot (-1) \cdot 1 = -0.0838
$$

  * **Bias update**:

$$
b = b + \eta \cdot \text{error} = -0.1 + 0.1 \cdot (-1) = -0.2
$$
  
* **Error Count after Sample 3**: 2 errors.

#### **Sample 4** (Epoch 2):

* **Input**: `[1, 1]`, Target: `1`
* **Prediction**: `0` (since `net_input ≈ -0.0899`).
* **Error**: `1 - 0 = +1` (there is an error).
* **Update**: Update weights and bias:

  * **Weight update**:

$$
w_0 = w_0 + \eta \cdot \text{error} \cdot x_0 = -0.0838 + 0.1 \cdot 1 \cdot 1 = 0.1162
$$

  * **Bias update**:

$$
b = b + \eta \cdot \text{error} = -0.2 + 0.1 \cdot 1 = -0.1
$$
  
* **Error Count after Sample 4**: 3 errors.

#### **Sample 5** (Epoch 2):

* **Input**: `[2, 2]`, Target: `1`
* **Prediction**: `1` (since `net_input ≈ 0.6202`).
* **Error**: `1 - 1 = 0` (no error).
* **Update**: No update needed.
* **Error Count after Sample 5**: 3 errors.

#### **Epoch 2 Summary**:

* The total number of errors after **Epoch 2** is `[3]` (3 samples incorrectly classified).

---

### 🧠 **Weight Update Calculation**

The **weight update** follows the formula:

$$
w_j = w_j + \eta \cdot \text{error} \cdot x_j
$$

Where:

* `w_j` is the weight for the corresponding feature `x_j`.
* `η` is the learning rate.
* `error` is the difference between the predicted and true label (`target - prediction`).
* `x_j` is the feature value for the current sample.

---

### 🧑‍🏫 Example of Weight Update (Epoch 2, Sample 3)

For **Sample 3 (Epoch 2)**, we had:

* **Input**: `[1, 0]`, **Target**: `0`
* **Prediction**: `1`
* **Error**: `0 - 1 = -1`

The **update** for weight `w_0` is:

$$
w_0 = w_0 + \eta \cdot \text{error} \cdot x_0
= 0.0162 + 0.1 \cdot (-1) \cdot 1
= -0.0838
$$

For bias `b`:

$$
b = b + \eta \cdot \text{error} = -0.1 + 0.1 \cdot (-1) = -0.2
$$

The weights and bias get updated based on the error feedback from the predictions.


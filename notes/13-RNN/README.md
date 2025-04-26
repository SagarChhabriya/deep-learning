## RNN: Recurrent Neural Networks

### Why RNNs?

Artificial Neural Networks (ANNs) work well with fixed-size, non-sequential input such as tabular data. Convolutional Neural Networks (CNNs) are designed for grid-like data, such as images. But when the order of data matters, like in a sentence or a time series, we need something different. That’s where Recurrent Neural Networks (RNNs) come in. They’re built specifically to handle sequential data.

### Understanding the Need for Sequential Models

Take a simple tabular dataset with columns like IQ, Marks, and Gender. The order of these inputs doesn’t change the meaning of the data, so using an ANN makes sense. But now imagine we’re working with a sentence like:

"Hi, my name is Sagar Chhabriya."

If we randomly shuffle the words to "Chhabriya my name Sagar hi is," the meaning is completely lost. This is why sequence matters. RNNs were originally developed back in the 1980s, but they’ve gained popularity more recently with the rise of modern applications in natural language processing and time series analysis.

### Common Examples of Sequential Data

1. Text data (sentences, documents)
2. Time series data (stock prices, weather)
3. Speech or audio
4. DNA sequences

### Applications of RNNs

- Sentiment analysis
- Next word prediction
- Image captioning
- Machine translation
- Question answering systems

### How RNNs Process Input

Unlike ANNs, which take a fixed-size input all at once, RNNs process data step by step, maintaining a hidden state that carries information across time steps.

---

## Preparing Data for RNNs

### Integer Encoding

Before feeding text into an RNN, it’s common to convert each word into an integer. This is known as integer encoding. For example, for the sentence:

"Hi there how are you"

We might assign:

Hi = 1, there = 2, how = 3, are = 4, you = 5

Now the sentence becomes a list of numbers. Since sentences can be of different lengths, we use padding to ensure they all have the same shape.

Example:
- [1, 2] becomes [1, 2, 0]
- [3, 4, 5] stays [3, 4, 5]

### Embedding

An embedding is a dense vector representation of words. Instead of representing a word as just an integer, embeddings capture the meaning of the word in a continuous vector space. Words with similar meanings are placed closer together.

In deep learning frameworks like Keras, we use an Embedding layer that learns these dense vector representations during training.

---

## Types of RNN Architectures

1. **Many-to-One**  
   Input is a sequence, output is a single value.  
   Examples: Sentiment analysis, movie rating prediction.

2. **One-to-Many**  
   Input is a single item, output is a sequence.  
   Examples: Image captioning, music generation.

3. **Many-to-Many**  
   Input and output are both sequences.  
   There are two subtypes:
   - Same input and output length: Example, part-of-speech tagging
   - Different lengths: Example, machine translation

4. **One-to-One**  
   Input and output are both single items.  
   Example: Image classification.  
   This isn’t a typical use case for RNNs, since there’s no sequence involved.

---

## Challenges in Training RNNs

### Long-Term Dependencies

RNNs often struggle when the current output depends on something far back in the sequence. For example:

- "Sindhi is spoken in Sindh." → Here, the word "Sindh" depends on "Sindhi," which is close by. This is a short-term dependency.
- "I went there last month, but I couldn't enjoy it because I don't understand ____." → The blank refers back to "Sindhi," which occurred much earlier in the sentence. This is a long-term dependency, and RNNs tend to forget information from far back in the sequence.

### Gradient Problems

Training RNNs over long sequences can lead to vanishing or exploding gradients. This makes learning unstable and inefficient. These are the two key reasons why basic RNNs are not great at handling long-term dependencies.

---

## What’s Next

Up next would be understanding **Backpropagation Through Time (BPTT)**, which is how RNNs are trained over sequences, followed by exploring more advanced models like **LSTMs** and **GRUs** that are designed to solve these long-term dependency issues.


### Backpropagation Through Time (BPTT)

BPTT is an extension of the standard backpropagation algorithm, specifically designed for RNNs. In standard backpropagation, we update the weights based on the error in a single forward pass. However, since RNNs process sequences step by step, the error depends not just on the current output but also on previous time steps. Here's a breakdown of the steps:

1. **Forward Pass**:
   - The RNN processes the input sequence step by step, calculating an output at each time step.
   - It maintains a hidden state, which is passed along from one time step to the next, capturing information from previous steps.

2. **Compute Loss**:
   - After processing all time steps, a loss function is calculated based on the output sequence (e.g., mean squared error for regression or cross-entropy for classification).

3. **Backward Pass**:
   - The key idea of BPTT is that the error at each time step must be propagated back through the network. This means that we need to compute gradients at each time step and update the weights accordingly.
   - Since the network maintains a hidden state that passes information from previous time steps, we have to propagate the error backward through time to update the hidden state and weights.

4. **Weight Updates**:
   - After the backward pass, weights are updated using gradient descent (or a variant like Adam).
   - This update happens for each time step and the weights are adjusted based on the error at each step.

### RNN Diagram: Basic Flow

Here’s a basic diagram that shows the flow of data in a simple RNN:

```
Input Sequence: x1, x2, x3, ..., xn

    +-------+      +-------+      +-------+      +-------+
    |       |      |       |      |       |      |       |
    |   x1  | ---> |  RNN  | ---> |  RNN  | ---> |  RNN  | ---> ... ---> Output
    |       |      |       |      |       |      |       |
    +-------+      +-------+      +-------+      +-------+
         ^              ^              ^              ^
         |              |              |              |
         +-- Hidden state and weights flow back
         (Backpropagation)
```

### **Backpropagation in RNN: Steps**

1. **Forward Pass**: For each time step, the input is passed into the RNN unit, the hidden state is updated, and the output is generated.
2. **Loss Calculation**: At the final step, the output is compared to the target, and the loss is computed.
3. **Backward Pass**: Gradients are calculated by applying the chain rule to propagate errors back in time (this is where "through time" comes into play).
4. **Weight Update**: After computing the gradients for each time step, weights are updated for each layer of the RNN, starting from the output layer and moving backward.

---

### RNN Types Overview: Diagram

Here's a quick breakdown of the different types of RNN architectures:

1. **Many-to-One** (e.g., Sentiment Analysis)
```
Input Sequence: x1, x2, x3, ..., xn --> [RNN Layers] --> Output: 1/0 or continuous value
```

2. **One-to-Many** (e.g., Image Captioning)
```
Input: Single image --> [RNN Layers] --> Output Sequence: Caption words
```

3. **Many-to-Many** (e.g., Machine Translation)
   - **Same length input-output** (POS Tagging)
   ```
   Input Sequence: x1, x2, x3, ..., xn --> [RNN Layers] --> Output Sequence: y1, y2, y3, ..., yn
   ```
   - **Different length input-output** (Machine Translation)
   ```
   Input Sequence: x1, x2, x3, ..., xn --> [RNN Layers] --> Output Sequence: y1, y2, y3, ..., ym
   ```

---

Great! Let's dive deeper into **LSTMs** (Long Short-Term Memory units) and **GRUs** (Gated Recurrent Units), which are both advanced versions of RNNs designed to handle the problems with basic RNNs, especially the issue of long-term dependencies.

---

## LSTM (Long Short-Term Memory)

### Motivation for LSTM

The main issue with basic RNNs is that they struggle to remember information over long sequences due to the **vanishing gradient problem**. This happens when the gradients, used to update the weights during backpropagation, get smaller as they propagate backward through many time steps, eventually becoming so small that they can't effectively update the weights. LSTMs were specifically designed to address this problem.

### How LSTM Works

LSTMs use a special structure called **gates** to control the flow of information. These gates allow the network to **remember** information for longer periods, and **forget** irrelevant data. Here's the key architecture:

1. **Cell State (C)**: The memory of the network, which carries information across the time steps.
2. **Hidden State (h)**: The output of the LSTM unit, which is passed to the next time step and can also be used as the output of the LSTM layer.
3. **Gates**:
   - **Forget Gate**: Decides what proportion of the cell state should be forgotten.
   - **Input Gate**: Decides what new information from the current input and previous hidden state should be added to the cell state.
   - **Output Gate**: Decides what part of the cell state will be output to the hidden state.

### Diagram of LSTM

```
     +------------+
     |  Forget    |
     |   Gate     | <---+
     +------------+     |
           |            |
           v            |
     +------------+      |
     |  Input     |      |
     |   Gate     | <----+
     +------------+      |
           |            |
           v            |
     +------------+      |   +-------------+
     |  Cell      |      |   | Output Gate |
     |  State     | ---->+-->|     h       |
     +------------+          +-------------+
           |                        ^
           v                        |
    +-------------+           +------------+
    |  Hidden     | <---------|  h (Hidden)|
    |   State     |           |   state    |
    +-------------+           +------------+
```

### LSTM Key Components:
1. **Forget Gate**:  
   It decides what information from the cell state should be thrown away. It outputs a number between 0 and 1 for each number in the cell state $C_{t-1}$. A 1 means "keep this" and 0 means "forget this."

2. **Input Gate**:  
   It updates the cell state with new information. First, it decides which values to update using a sigmoid activation, then a tanh layer creates candidate values to add to the state.

3. **Cell State Update**:  
   The cell state is updated by adding the input gate's output to the forget gate’s output.

4. **Output Gate**:  
   The output gate filters the cell state and produces the hidden state $h_t$. This decides what the next hidden state will be, which is also the output of the LSTM unit.

---

## GRU (Gated Recurrent Unit)

### Motivation for GRU

GRUs are a simpler, more computationally efficient alternative to LSTMs. They were introduced to address the vanishing gradient problem while keeping the model simpler and faster to train. GRUs combine some of the components of LSTMs and are easier to implement, often providing comparable performance.

### How GRU Works

GRUs merge the forget and input gates into a single **update gate**, which simplifies the model. GRUs only have two gates:

1. **Update Gate**: Determines how much of the previous hidden state needs to be carried over and how much of the current input should be added to the state.
2. **Reset Gate**: Controls how much of the previous state should be forgotten when computing the current hidden state.

### GRU Structure:

- **Update Gate**: Decides how much of the past information (previous hidden state) should be kept.
- **Reset Gate**: Decides how much of the past hidden state should be ignored when calculating the current hidden state.

### Diagram of GRU

Here’s a simplified diagram of a GRU unit:

```
      +------------+
      |  Reset     |
      |   Gate     | <---+
      +------------+     |
            |            |
            v            |
      +------------+      |      +-------------+
      | Update     |      |      |   Hidden    |
      | Gate       | ---->+----->|   State     |
      +------------+            +-------------+
            |                           ^
            v                           |
      +-------------+            +-------------+
      |  Current    | <----------|  h (Hidden) |
      |  Hidden     |            |   State     |
      +-------------+            +-------------+
```

---

## Key Differences Between LSTM and GRU

1. **Number of Gates**:
   - LSTMs have three gates: forget, input, and output.
   - GRUs combine the forget and input gates into a single update gate, and have a reset gate.

2. **Memory and Output**:
   - LSTMs use both the hidden state and cell state, allowing them to store more detailed information.
   - GRUs only use the hidden state and don’t have a separate cell state, making them simpler and faster.

3. **Computational Complexity**:
   - LSTMs are more complex due to the additional gate (the output gate), which can make them slower to train.
   - GRUs are simpler and therefore faster, often providing comparable performance with fewer parameters.

4. **Performance**:
   - In some cases, LSTMs perform better, especially in tasks with very long sequences.
   - GRUs are often preferred when computational efficiency is more important, or when sequences are not extremely long.

---

## Summary

- **LSTMs** are great for tasks where long-term dependencies are crucial, such as language modeling, speech recognition, and time series forecasting.
- **GRUs** are a simpler alternative to LSTMs and work well for many tasks where computational efficiency is important, without a significant loss in performance.

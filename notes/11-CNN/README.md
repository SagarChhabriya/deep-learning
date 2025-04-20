
# CNN (Convolutional Neural Network)

A **Convolutional Neural Network (CNN)** is a type of neural network specifically designed to process grid-like data structures, such as images (2D) or time series data (1D). It is particularly effective for tasks such as image recognition, classification, and object detection.

### Key Differences between CNN and ANN

- **ANN (Artificial Neural Network)**: Uses matrix multiplication for processing input data.
- **CNN**: Uses convolutional operations to extract features from the data.

A typical CNN consists of:
1. **Convolutional Layer**
2. **Pooling Layer**
3. **Fully Connected (FC) Layer**

### Why Not Use ANN for Image Data?

- **High Computation Cost**: Using ANN for image data can lead to very high computational costs.
- **Overfitting**: ANN is prone to overfitting, especially with image data.
- **Loss of Spatial Information**: ANN loses important spatial information (like the arrangement of pixels), which is critical in images.

For example, if you have a 40 x 40 image and 50 nodes in the hidden layer, the number of parameters in a fully connected layer will be 40 x 40 x 50 = 80,000. This makes ANN inefficient for large images.

### Convolutional Layers: Feature Extraction

Convolutional layers act as filters to extract important features from the image, such as edges, textures, and shapes. This is similar to how the human visual cortex processes images.

- **Visual Cortex**:
  - **Simple Cells**: Detect specific features, like orientation.
  - **Complex Cells**: Detect more complex features such as movement and depth.

### CNN vs Visual Cortex

An analogy is the human brain's visual cortex, which detects simple features like edges in the early stages and more complex patterns like shapes and objects later on. In CNNs, the initial layers focus on detecting simple features, while the deeper layers combine these features to identify more complex structures.

### Convolution Operation in CNN

CNNs use convolutional operations to detect patterns in images. A **kernel** (or filter) slides over the image to compute the feature map.

#### Example: Edge Detection

- **Vertical Edges + Horizontal Edges**: For example, detecting edges in a car image.
- **Edges** = Intensity: Edges in an image correspond to significant changes in intensity values.
- **Filter/Kernel Matrix Multiplication**: The process of applying a kernel to an image to detect specific features like edges.

In this example, the kernel is used to detect horizontal edges in an image of a car. The original image matrix is multiplied with a kernel (filter), and the result is the feature map.

- **Original Image**:

    ```
    [0   0   0   0   0   0]
    [0   0   0   0   0   0]
    [0   0   0   0   0   0]
    [255 255 255 255 255 255]
    [255 255 255 255 255 255]
    [255 255 255 255 255 255]
    ```

- **Kernel for Horizontal Edge Detection**:

    ```
    [-1  -1  -1]
    [ 0   0   0]
    [ 1   1   1]
    ```

- **Feature Map**: The result of the convolution is a feature map, which highlights horizontal edges.

```css
            [ 0   0   0   0   0   0 ]
            [ 0   0   0   0   0   0 ]
            [ 0   0   0   0   0   0 ]   *     [ -1  -1  -1 ]    =   Feature Map
            [255 255 255 255 255 255]         [  0   0   0 ]
            [255 255 255 255 255 255]         [  1   1   1 ]
            [255 255 255 255 255 255]
```

Scientists have designed various kernels to detect different types of edges, such as top edges, bottom edges, left edges, and right edges.

In deep learning, you don't need to manually design these kernels. Instead, the network learns the optimal values for the kernels during training via backpropagation. Essentially, these kernel values are just weights that are adjusted to find the best features for the task at hand.

[deeplizard.com | Convol Operation Visualizer](https://deeplizard.com/resource/pavq7noze2)

- **Finding the size of feature map**
    - Image (28 x 28 ) * kernal (3 x 3) = Feature Map (26 x 26)
    - Image (n  x  n ) * kernal (m x m) = Feature Map (n - m +1 ) x (n - m + 1) 


- Working with RGB Images
In case of RGB the kernal becomes 3 x 3 x 3
![](../assets/25-cnn-rgb.png)


### Convolution Formula

To compute the feature map size after convolution:

$$
\text{Feature Map Size} = (n - m + 1) \times (n - m + 1)
$$

Where:
- $n$ is the size of the image,
- $m$ is the size of the kernel.

### Working with RGB Images

In RGB images, the kernel is 3D, with dimensions corresponding to the three color channels (Red, Green, Blue).

- **RGB Image Example**: An image of size 288 x 228 x 3 (height x width x channels).
- The kernel in this case will have dimensions $3 \times 3 \times 3$, where each of the 3 channels (RGB) is processed separately.

### Multiple Filters

CNNs use multiple filters to detect different features in the image. For example, one filter may detect vertical edges, while another detects horizontal edges.

$$
(m \times m \times c) * (n \times n \times c) = (m - n + 1) \times (m - n + 1) \times f
$$

Where:
- $c$ is the number of channels,
- $f$ is the number of filters.

### Padding and Strides in CNN

#### Padding

Padding is used to prevent the loss of important information during convolution. Without padding, the image size shrinks after each convolution operation.

- **Before padding**

![](../assets/26-before-padding.gif)

- **After Zero Padding**

![](../assets/27-after-zero-padding.gif)


- **Zero Padding**: Add extra rows and columns (usually filled with zeros) around the image before applying convolution. This helps preserve the original size of the image.

**Formula for padding**:

Before padding:

$$
\text{Feature Map Size} = n - m + 1
$$

After padding:

$$
\text{Feature Map Size} = \frac{n + 2p - m}{s} + 1
$$

Where:
- $p$ is the padding,
- $s$ is the stride.

In Keras:
- **Valid Padding**: No padding, the traditional method.
- **Same Padding**: Keras automatically finds the optimal padding.

#### Strides

Strides control how much the kernel moves over the image. The stride determines how much the filter shifts during each operation.

- A stride of $(1, 1)$ means the kernel moves one step at a time in both horizontal and vertical directions.
- Larger strides reduce the feature map size, making the model more computationally efficient but potentially losing some details.

**Stride Formula**:

$$
\text{Feature Map Size} = \frac{n - m}{s} + 1
$$

Where $s$ is the stride size.

### Pooling Layer

The **Pooling Layer** reduces the size of the feature map and retains important information. It helps in reducing computation and preventing overfitting. There are two types of pooling:
- **Max Pooling**: Takes the maximum value from a patch of the feature map.
- **Average Pooling**: Takes the average value from a patch of the feature map.

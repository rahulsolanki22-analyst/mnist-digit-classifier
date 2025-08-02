# MNIST Digit Classifier from Scratch (Using NumPy)

This project implements a basic neural network from scratch (no libraries like TensorFlow or scikit-learn) to classify handwritten digits using the MNIST dataset. It is designed to show how a neural network actually works internally.

---

##  Project Structure

```
mnist-digit-classifier/
├── mnist-ml.ipynb      # Jupyter notebook with complete neural network code
├── train.csv           # Training data in CSV format
└── README.md           # Project overview and instructions
```

---

##  Objective

To train a neural network manually (using only NumPy) to recognize handwritten digits (0–9) from the MNIST dataset.

---

##  Technologies Used

- **Python 3**
- **NumPy** – for array operations and matrix calculations
- **Matplotlib** – to visualize accuracy/loss
- **Pandas** – for CSV data loading
- **Jupyter Notebook** – for interactive experimentation

---

##  Neural Network Architecture

- Input Layer: 784 neurons (28x28 pixel image)
- Hidden Layer: 10 neurons (with ReLU activation)
- Output Layer: 10 neurons (for digits 0–9, with Softmax)

**Training Method**:
- Manual forward propagation
- Manual backward propagation
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD)

---

##  Features

- Manual data preprocessing (normalization, train/dev split)
- Weight initialization with He strategy
- Implemented ReLU and Softmax activation from scratch
- Backpropagation for gradient updates
- Accuracy evaluation and plotting

---

## 📊 Dataset Info

- `train.csv` contains:
  - 42,000+ rows of handwritten digit images
  - Each image is 28x28 pixels, flattened into 784 values
  - First column is the label (digit), remaining 784 are pixel intensities

---

##  How to Run

1. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib
   ```

2. **Open the notebook**
   ```bash
   jupyter notebook mnist-ml.ipynb
   ```

3. **Run all cells**
   - It will train the neural network
   - Print training accuracy
   - Optionally plot performance metrics

---

## Sample Output

- Accuracy after training: ~82%–88% (depending on random weight init and learning rate)
- Training visualization (accuracy/loss graph)

---

##  Notes

- This project is purely educational and avoids high-level libraries.
- Designed to help understand the inner workings of a neural network.
- Optimizations like mini-batch gradient descent, momentum, and regularization are **not** included.

---




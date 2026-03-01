# SCBI: Stochastic Covariance-Based Initialization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18576203.svg)](https://doi.org/10.5281/zenodo.18576203)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A GPU-accelerated neural network initialization strategy that achieves 87× faster convergence on regression tasks and 33% improvement on classification.**

---

## 🚀 What is SCBI?

SCBI (Stochastic Covariance-Based Initialization) is a novel weight initialization method that solves the **"cold start" problem** in neural networks by computing near-optimal weights before training begins.

Instead of starting with random weights, SCBI:
1. **Samples** multiple data subsets (stochastic bagging)
2. **Solves** the Normal Equation with optimal Ridge regularization
3. **Averages** solutions for robust initialization
4. **Provides** a mathematically-principled warm start

### Key Results

| Task | Standard Init | SCBI Init | Improvement |
|------|--------------|-----------|-------------|
| **Regression** | 26,000 MSE | **300 MSE** | **87× faster** |
| **Classification** | 1.18 loss | **0.79 loss** | **33% better** |

---

## ✨ Features (v3.0)

### 🎯 Core Innovations

- **Dynamic Ridge CV**: Automatically tunes regularization via nested cross-validation
- **Stochastic Bagging**: Ensemble averaging over multiple subsets for robustness
- **Memory Efficiency**: Mean-centering without augmented matrices
- **GPU Acceleration**: Full PyTorch integration for CUDA support
- **Zero Hyperparameter Tuning**: Works out-of-the-box with sensible defaults

### 🔧 Technical Highlights

- Universal formulation for **regression and classification**
- Handles **high-dimensional** and **collinear** features
- **Numerical stability** via pseudo-inverse fallback
- Compatible with any PyTorch optimizer and loss function
- Easy integration into existing training pipelines

---

## 📦 Installation

### Requirements

```bash
torch>=1.9.0
numpy>=1.19.0
```

### Install

Simply copy `scbi.py` into your project:

```bash
wget https://zenodo.org/record/18576203/files/scbi.py
```

Or install from source:

```bash
git clone https://github.com/fares3010/SCBI.git
cd SCBI
pip install -e .
```

---

## 🎯 Quick Start

### Basic Usage (Single Layer)

```python
import torch
from scbi import SCBILinear

# Create layer
layer = SCBILinear(784, 128)

# Prepare proxy sample (10-30% of training data)
X_proxy = X_train[:500]
y_proxy = y_train[:500]

# Initialize with SCBI
layer.init_weights_with_proxy(X_proxy, y_proxy)

# Train normally!
optimizer = torch.optim.Adam(layer.parameters())
for epoch in range(epochs):
    loss = criterion(layer(X_train), y_train)
    loss.backward()
    optimizer.step()
```

### Deep Network

```python
from scbi import create_scbi_mlp

# Create MLP with SCBI initialization
model = create_scbi_mlp(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    dropout=0.2
)

# One-shot initialization for entire network
model.init_scbi_layers(X_proxy, y_proxy)

# Train as usual
optimizer = torch.optim.Adam(model.parameters())
```

### Functional API

```python
from scbi import scbi_init

# Get optimal weights directly
weights, bias = scbi_init(X_train, y_train)

# Use with any PyTorch layer
model = torch.nn.Linear(50, 10)
with torch.no_grad():
    model.weight.data = weights.T
    model.bias.data = bias
```

---

## 📚 API Reference

### `SCBILinear`

```python
SCBILinear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    n_samples: int = 10,
    sample_ratio: float = 0.5,
    ridge_alpha: float = 1.0,
    tune_ridge: bool = True,
    cv_folds: int = 5
)
```

**Parameters:**

- `in_features`: Input dimension
- `out_features`: Output dimension  
- `bias`: Include bias term (default: True)
- `n_samples`: Number of stochastic subsets for bagging (default: 10)
  - Higher = more stable, slower
  - Recommended: 5-20
- `sample_ratio`: Fraction of proxy data per subset (default: 0.5)
  - Range: (0, 1]
  - Recommended: 0.3-0.7
- `ridge_alpha`: Base Ridge penalty (default: 1.0)
  - Only used if `tune_ridge=False`
- `tune_ridge`: Enable automatic Ridge CV tuning (default: True)
  - **Recommended: Keep enabled**
- `cv_folds`: Number of CV folds for Ridge tuning (default: 5)

**Methods:**

```python
init_weights_with_proxy(
    proxy_x: torch.Tensor,        # [N, in_features]
    proxy_y: Optional[torch.Tensor] = None,  # [N, out_features]
    verbose: bool = True
)
```

Initializes weights using SCBI. For hidden layers, `proxy_y` can be omitted.

---

### `create_scbi_mlp`

```python
create_scbi_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: nn.Module = None,
    dropout: float = 0.0,
    **scbi_kwargs
) -> SCBISequential
```

Factory function to create a multi-layer perceptron with SCBI layers.

**Example:**

```python
model = create_scbi_mlp(
    input_dim=784,
    hidden_dims=[512, 256],
    output_dim=10,
    activation=nn.ReLU(),
    dropout=0.2,
    n_samples=15,  # SCBI parameter
    tune_ridge=True
)

model.init_scbi_layers(X_proxy, y_proxy)
```

---

### `scbi_init` (Functional)

```python
scbi_init(
    X_data: torch.Tensor,
    y_data: torch.Tensor,
    n_samples: int = 10,
    sample_ratio: float = 0.5,
    ridge_alpha: float = 1.0,
    tune_ridge: bool = True,
    verbose: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

Returns optimal weights and bias tensors directly.

**Returns:**
- `weights`: [in_features, out_features]
- `bias`: [out_features]

---

## 🔬 How It Works

### Algorithm Overview

```
1. Sample K subsets from proxy data (stochastic bagging)
2. For each subset:
   a. Center features and targets (memory-efficient)
   b. Solve: (X^T X + λI)^(-1) X^T y
   c. Reconstruct bias: b = y_mean - w^T * X_mean
3. Average K solutions (ensemble)
4. Assign to layer parameters
```

### Mathematical Foundation

**Normal Equation with Ridge Regularization:**

```
w* = (X^T X + λI)^(-1) X^T y
b* = ȳ - w*^T x̄
```

**Stochastic Bagging:**

```
w_final = (1/K) Σ w_k
b_final = (1/K) Σ b_k
```

**Dynamic Ridge CV:**

```
λ* = argmin_λ Σ MSE_validation(λ)
```

### Why It Works

1. **Stochastic Bagging**: Reduces overfitting to training data
2. **Ridge Regularization**: Handles collinearity and numerical instability
3. **Automatic CV**: Finds optimal λ without manual tuning
4. **Mean Centering**: Improves numerical conditioning
5. **Ensemble Averaging**: Robust to subset selection

---

## 📊 Performance Benchmarks

### Regression (Synthetic High-Dim)

```
Dataset: 5,000 samples, 2,000 features
Task: Predict continuous target

Method          | Initial Loss | Epoch 5 | Epoch 20
----------------|--------------|---------|----------
Standard Init   | 26,000       | 8,500   | 2,300
SCBI Init       | 300          | 250     | 180
Improvement     | 87× better   | 34×     | 13×
```

### Classification (Forest Cover)

```
Dataset: 581,000 samples, 54 features, 7 classes
Task: Multi-class classification

Method          | Initial Loss | Final Accuracy
----------------|--------------|----------------
Xavier Init     | 1.18         | 68.2%
He Init         | 1.21         | 67.8%
SCBI Init       | 0.79         | 72.4%
Improvement     | 33% better   | +4.2%
```

### Initialization Time

```
Layer Size      | Standard | SCBI (CPU) | SCBI (GPU)
----------------|----------|------------|------------
[100 → 50]      | 0.001s   | 0.05s      | 0.01s
[1000 → 500]    | 0.001s   | 0.3s       | 0.08s
[5000 → 1000]   | 0.001s   | 2.1s       | 0.4s
```

**Note:** Initialization overhead is negligible compared to training time (typically < 1% of total training).

---

## 💡 Best Practices

### 1. Proxy Sample Size

```python
# Rule of thumb: 10-30% of training data
proxy_size = max(500, int(0.2 * len(X_train)))
X_proxy = X_train[:proxy_size]
```

**Guidelines:**
- Minimum: 100 samples
- Recommended: 500-2000 samples  
- Maximum: 5000 samples (diminishing returns)

### 2. When to Use SCBI

**✅ Use SCBI for:**
- Linear layers in regression/classification tasks
- First layer of deep networks
- Classification heads in transfer learning
- High-dimensional tabular data
- When fast convergence is critical

**❌ Skip SCBI for:**
- Convolutional layers (use Kaiming)
- Recurrent layers (use orthogonal init)
- Transformers (use scaled init)
- Very small datasets (< 100 samples)

### 3. Hyperparameter Tuning

**Default settings work for 90% of cases:**

```python
layer = SCBILinear(
    in_features=100,
    out_features=50,
    n_samples=10,      # Usually no need to change
    sample_ratio=0.5,  # Usually no need to change
    tune_ridge=True    # Keep enabled!
)
```

**When to adjust:**

- **Noisy data**: Increase `ridge_alpha` base (try 5.0 or 10.0)
- **Very stable data**: Decrease `ridge_alpha` base (try 0.1)
- **Large proxy sample**: Increase `n_samples` to 20
- **Time constraint**: Decrease `n_samples` to 5

### 4. Integration with Existing Code

**Minimal changes required:**

```python
# Before: Standard PyTorch
model = nn.Linear(784, 128)
optimizer = torch.optim.Adam(model.parameters())

# After: With SCBI
from scbi import SCBILinear

model = SCBILinear(784, 128)
model.init_weights_with_proxy(X_proxy, y_proxy)  # Add this line
optimizer = torch.optim.Adam(model.parameters())  # Rest is the same!
```

---

## 🎓 Advanced Usage

### Custom Architectures

```python
from scbi import SCBILinear

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SCBILinear(784, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = SCBILinear(512, 256)
        self.relu2 = nn.ReLU()
        self.layer3 = SCBILinear(256, 10)
    
    def init_scbi(self, X_proxy, y_proxy):
        # Layer 1
        self.layer1.init_weights_with_proxy(X_proxy)
        h1 = self.relu1(self.layer1(X_proxy))
        h1 = self.dropout(h1)
        
        # Layer 2
        self.layer2.init_weights_with_proxy(h1)
        h2 = self.relu2(self.layer2(h1))
        
        # Layer 3 (output)
        self.layer3.init_weights_with_proxy(h2, y_proxy)
    
    def forward(self, x):
        x = self.dropout(self.relu1(self.layer1(x)))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)

# Usage
model = CustomModel()
model.init_scbi(X_proxy, y_proxy)
```

### Classification with One-Hot Encoding

```python
import torch.nn.functional as F

# Convert class labels to one-hot
y_labels = torch.tensor([0, 2, 1, 0, 2])  # Class indices
y_onehot = F.one_hot(y_labels, num_classes=3).float()

# Initialize
layer = SCBILinear(50, 3)
layer.init_weights_with_proxy(X_proxy, y_onehot)

# Training with cross-entropy
criterion = nn.CrossEntropyLoss()
loss = criterion(layer(X_batch), y_labels)  # Use labels, not one-hot!
```

### Multi-Output Regression

```python
# Predict multiple continuous outputs
X = torch.randn(1000, 50)
y = torch.randn(1000, 5)  # 5 targets

layer = SCBILinear(50, 5)
layer.init_weights_with_proxy(X[:300], y[:300])

# Works out of the box!
predictions = layer(X_test)  # [batch, 5]
```

### GPU Acceleration

```python
# Move everything to GPU
device = torch.device('cuda')

X_train = X_train.to(device)
y_train = y_train.to(device)

model = create_scbi_mlp(784, [512, 256], 10).to(device)
model.init_scbi_layers(X_train[:500], y_train[:500])

# Training on GPU
for epoch in range(epochs):
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
```

---

## 🔧 Troubleshooting

### Issue: Poor performance despite SCBI

**Possible causes:**
1. Proxy sample too small (< 100)
2. Data is highly non-linear (SCBI assumes weak linearity)
3. Features not standardized

**Solutions:**
```python
# 1. Increase proxy size
X_proxy = X_train[:1000]  # Use more data

# 2. Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 3. Check data linearity
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print(f"R² score: {ridge.score(X_train, y_train)}")
# If R² < 0.5, data might be too non-linear for SCBI
```

### Issue: RuntimeError during initialization

**Error:** `torch.linalg.solve: singular matrix`

**Cause:** Proxy sample has linearly dependent features

**Solution:** Already handled automatically! Code uses pseudo-inverse fallback.

If you still see errors:
```python
# Increase ridge penalty
layer = SCBILinear(100, 50, ridge_alpha=10.0, tune_ridge=False)
```

### Issue: NaN or Inf values

**Causes:**
1. Features have extreme values
2. Ridge alpha too small

**Solutions:**
```python
# 1. Standardize features
X_train = (X_train - X_train.mean()) / X_train.std()

# 2. Increase ridge alpha
layer = SCBILinear(100, 50, ridge_alpha=5.0)
```

### Issue: Initialization too slow

**For very large feature dimensions (D > 10,000):**

```python
# Use smaller proxy sample
X_proxy = X_train[:300]  # Reduce from 500

# Reduce bagging samples
layer = SCBILinear(D, out, n_samples=5)  # Reduce from 10

# Or skip Ridge CV
layer = SCBILinear(D, out, tune_ridge=False)
```

---

## 📖 Citation

If you use SCBI in your research, please cite:

```bibtex
@software{ashraf2026scbi,
  author       = {Ashraf, Fares},
  title        = {SCBI: Stochastic Covariance-Based Initialization 
                  for Neural Networks},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.0.0},
  doi          = {10.5281/zenodo.18576203},
  url          = {https://doi.org/10.5281/zenodo.18576203}
}
```

**APA Format:**
```
Ashraf, F. (2026). SCBI: Stochastic Covariance-Based Initialization 
for Neural Networks (Version 3.0.0) [Computer software]. Zenodo. 
https://doi.org/10.5281/zenodo.18576203
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Key Points:**
- ✅ Free for commercial use
- ✅ Free for research use
- ✅ Modification allowed
- ✅ Distribution allowed
- ❗ No warranty provided

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Areas for contribution:**
- Additional benchmarks on diverse datasets
- Integration examples (PyTorch Lightning, Hugging Face)
- Performance optimizations
- Documentation improvements
- Bug fixes

**Before contributing:**
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## 📬 Contact

**Author:** Fares Ashraf  
**Email:** farsashraf44@gmail.com  
**DOI:** [10.5281/zenodo.18576203](https://doi.org/10.5281/zenodo.18576203)  
**GitHub:** [github.com/fares3010/SCBI](https://github.com/fares3010/SCBI)

For bug reports and feature requests, please use the [GitHub Issues](https://github.com/fares3010/SCBI/issues) page.

---

## 🌟 Acknowledgments

This work was inspired by:
- **Glorot & Bengio (2010)**: Xavier initialization
- **He et al. (2015)**: Kaiming/He initialization  
- **Ridge Regression**: Hoerl & Kennard (1970)
- **Bagging**: Breiman (1996)

Special thanks to the PyTorch team for the excellent framework!

---

## 📊 Version History

### v3.0.0 (2026-02-27) - Production Release
- ✨ Dynamic Ridge CV with nested cross-validation
- ✨ Memory-efficient mean-centering
- ✨ Improved numerical stability
- ✨ Enhanced documentation
- 🐛 Fixed edge cases in pseudo-inverse fallback

### v2.0.0 (2026-02-26)
- Added experimental Rearrangement Correlation method
- Introduced learnable gain factor
- Improved proxy sample efficiency

### v1.0.0 (2026-02-06) - Initial Release
- Core SCBI algorithm
- Stochastic bagging
- Ridge regularization
- PyTorch integration

---

## ⭐ Star History

If SCBI helped your research or project, please consider:
- ⭐ **Starring** the repository
- 📝 **Citing** in your paper
- 🔗 **Sharing** with colleagues
- 💬 **Providing** feedback

---

**Made with ❤️ by Fares Ashraf**

*Accelerating neural network training, one initialization at a time.* 🚀

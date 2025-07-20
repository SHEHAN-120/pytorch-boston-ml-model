# Boston Housing Price Prediction using PyTorch

This project implements a simple feedforward neural network (FFNN) in PyTorch to predict house prices from the Boston Housing dataset.

## 📊 Dataset

The [Boston Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) contains 506 samples with 13 features, including crime rate, number of rooms, property tax rate, and more.

## 🚀 Model Overview

* Framework: PyTorch
* Model: Feedforward Neural Network (2 linear layers + ReLU)
* Loss: Mean Squared Error (MSE)
* Optimizer: SGD (Stochastic Gradient Descent)
* Input features: 13
* Output: Predicted house price

## 📆 Technologies Used

* Python 3.x
* PyTorch
* scikit-learn
* NumPy
* Matplotlib

## 📃 Tools Used

* Google Colab
* Jupyter Notebook environment
* matplotlib for visualization
* torch for model training
* sklearn for data preprocessing and dataset loading

## 🚧 Challenges Faced

* Properly scaling and reshaping the dataset tensors
* Avoiding common PyTorch mistakes
* Handling data type conversions between NumPy and PyTorch
* Plotting predictions with correctly shaped arrays

## 📆 Training

The training loop runs for 100 epochs and logs the loss every 10 epochs.

### Loss Plot

The loss over epochs is plotted using Matplotlib.


## 📊 Predictions

After training, the model is evaluated on the test set and predictions are visualized:

```python
plt.scatter(X_test.numpy().squeeze(), y_test.numpy(), color='red', label='Actual')
plt.plot(X_test.numpy().squeeze(), predictions.numpy().squeeze(), color='blue', label='Predicted')
```



## ✅ Output Example

```text
Epoch [10/100], Loss: 78.1234
Epoch [20/100], Loss: 61.0123
...
Epoch [100/100], Loss: 21.7890
```

## 📜 License

This project is open-source and available under the MIT License.

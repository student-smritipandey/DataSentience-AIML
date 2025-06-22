# BIDIRECTIONAL LSTM

Bi-directional long-short term memory(bi-LSTM) is the type of RNN which makes the input sequence information flow in both directions forwards as well as backwards. It helps us preserve the future information as well as the past.

## Use Cases of Bi-Directional LSTM

Bidirectional LSTM is used in cases where flow of information from backward and forward layers makes execution of sequence-to-sequence tasks easier.Some examples are as follows:

- Text classification
- Speech Recognition
- Forecasting Models
- Natural Language Processing
- Dependency Parsing

## Working of Bi-Directional LSTM

- Bi- directional LSTM runs the inputs in two ways, one from past to future and one from future to past.
- This variation in approach from unidirectional LSTM helps to preserve information from the future as well as past.
- This architecture allows the neural networks to have both backward and forward information about the sequence at every time step.
- Forward Pass:
  - Forward states (from t = 1 to N) and backward states (from t = N to 1) are passed.
  - Output neuron values are passed (from t = 1 to N).
- Backward Pass:
  - Output neuron values are passed (t = N to 1).
  - Forward states (from t = N to 1) and backward states (from t = 1 to N) are passed.

Both the forward and backward passes together train a Bidirectional LSTM .

## Advantage of Bi-Directional LSTM

Bi-LSTMs effectively increase the amount of information available to the network, improving the context available to the model.

## When to avoid Bi-Directional LSTM

- One limitation with Bidirectional LSTM is that the entire sequence must be available before we can make predictions.
- In applications such as real-time speech recognition, the entire utterance may not be available before predictions and Bi-Directional LSTMs should be avoided in such cases.
- Stacking many layers of Bi-directional LSTM creates the vanishing gradient problem.So, it should be avoided in case of Deep Stacked layers of LSTM as well.

## Example Use Case of Bi-Directional LSTM

- This project demonstrates the prediction of the temperature for the next 30 days by the neural network model.
- We are using Bidirectional LSTM Model in RNN.

## Project Files

### Models
- **temperature-prediction-using-bidirectional-lstm.ipynb** - Original temperature prediction implementation
- **bidirectional-lstm.ipynb** - **Enhanced version with advanced features and optimizations**
- **my_best_model.epoch05-loss0.00.hdf5** - Trained model file with 99.99928% accuracy

### Dataset
- **testset.csv** - Test dataset for temperature prediction

## Key Improvements in Enhanced Version

### 🚀 **GPU Acceleration**
- **CUDA Support**: Full GPU acceleration with Tesla T4 GPUs
- **Multi-GPU Training**: Utilizes both GPU devices for faster training
- **Optimized Performance**: Significantly reduced training time with GPU parallelization

### 🧠 **Advanced Model Architecture**
- **Bidirectional LSTM Layers**: Enhanced with 32 units and return sequences
- **Batch Normalization**: Improved training stability and convergence
- **Dropout Regularization**: Multiple dropout layers (0.2) to prevent overfitting
- **Dense Layers**: Additional dense layers with ReLU activation for better feature learning

### ⚡ **Smart Training Optimizations**
- **Early Stopping**: Automatic training termination when validation loss plateaus (patience=5)
- **Learning Rate Scheduling**: Adaptive learning rate reduction (ReduceLROnPlateau)
  - Monitors validation loss
  - Reduces learning rate by 50% when plateau detected
  - Minimum learning rate of 1e-6
- **Model Checkpointing**: Saves best model based on validation loss
- **Larger Batch Size**: Increased to 128 for better GPU utilization

### 📊 **Enhanced Data Processing**
- **Larger Dataset**: Training on 75,136 samples (vs smaller original dataset)
- **Better Data Scaling**: MinMaxScaler for optimal LSTM performance
- **Time Series Window**: 100-step lookback window for sequence prediction
- **Robust Validation**: 20% validation split for reliable model evaluation

### 🎯 **Performance Metrics**
- **Faster Training**: ~20ms/step with GPU acceleration
- **Improved Convergence**: Better loss reduction patterns
- **Enhanced Predictions**: More accurate temperature forecasting
- **30-Day Forecasting**: Extended prediction capabilities

## Setup instructions

### Prerequisites

you need to install all the necessary libraries in order to run the project
`pandas`
`sklearn`
`numpy`
`tensorflow`
`keras`

### Installing

```
pip install pandas
pip install tensorflow
pip install numpy
pip install sklearn
pip install keras
```

### GPU Requirements (Optional but Recommended)
- **CUDA-compatible GPU** (Tesla T4, RTX series, etc.)
- **CUDA Toolkit** (version 11.0 or higher)
- **cuDNN** (compatible version)

## Model Performance

The enhanced model achieves:
- **Training Loss**: ~0.0009 (significantly improved)
- **Validation Loss**: ~0.0010 (stable and low)
- **Training Speed**: ~20ms/step with GPU acceleration
- **Convergence**: Faster and more stable training

## Author(s)

**Current Maintainer**: [Ashutosh Singh] (https://github.com/AshutoshSingh058)
**Original Author**: [Shreya Ghosh](https://github.com/shreay024)

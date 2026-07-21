# Stock Prediction Neural Network and Machine Learning Examples (Python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Stock Prediction Neural Network and Machine Learning Examples ](https://repository-images.githubusercontent.com/669594930/1b661bb8-d5d8-40ad-9d94-c3084f3df2fc)

## Contents:
* [Simple Examples](#simple-examples)
* [Hyperparameter Optimization](#hyperparameter-optimization)
* [Getting Started](#getting-started)
* [About the Example Stock Data](#about-the-example-stock-data)

## Overview
These are ML and NN methods ready to launch out of the box. Designed to be easy for those looking to learn new techniques for stock prediction. These examples are meant to be simple to understand and highlight the essential components of each method. Examples also show how to run the models on current data in order to get stock predictions.

> **Educational use only:** These examples are demonstrations, not investment
> advice or evidence of a profitable strategy. Classification scores do not
> account for slippage, liquidity, taxes, market impact, or regime changes.

### Machine Learning examples:
* Genetic algorithms
* Gradient boost
* K-means clustering
* Logistic regression
* Random Forest
* Support vector machines (SVM)

### Neural Net examples:
* Feed-forward neural networks (FFNN)
* Long short-term memory (LSTM)
* Recurrent Neural Networks (RNN)

### Neural Net library examples:
* Keras
* Lightning
* PyTorch
* Tensorflow

## Getting Started

1. **Clone this repository.**
2. **Navigate to the project directory.**
3. **Create a virtual environment and install the necessary libraries:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                 # classical ML examples
pip install -e '.[torch]'        # PyTorch and Lightning examples
pip install -e '.[tensorflow]'   # TensorFlow/Keras examples
```

Use `pip install -r requirements.txt` to install every framework.

4. **Download the starter data:**
   - These examples require precisely formatted stock data with the following properties:
     * **Windowed:** Time series is segmented into regular windows.
     * **Boolean labels:** Rather than predicting specific values at specific times, the data is classified.
     * **Test/Train split:** Split chronologically for no data overlaps and look-ahead bias.
   - [Download the starter data.](https://d.at/example-data) and save the `example_data` directory to this project folder.

5. **Run an example from the repository root:**

```bash
python -m simple_examples.machine_learning.logistic_regression
python -m simple_examples.neural_networks.pytorch_ffnn
```

## Neural Net Hyperparameter Optimization
Designed for easy configuration of what hyperparameter values are explored. Multi-threaded processing for quick runtimes.

1. Code is in `hyperparameter_tuning`
2. Edit `config.py` to suit your needs
3. Run `python -m hyperparameter_tuning.hyper_main` from the repository root

The tuner uses the chronological tail of `train.csv` for model selection and
does not inspect `test.csv`. Configure the validation fraction and embargo in
`hyperparameter_tuning/config.py`.

Hyperparameter readme here: [Hyperparameter Tuning](hyperparameter_tuning/README.md)

## About the Example Stock Data

This code can be run with the example stock data available at [D.AT example data](https://d.at/example-data).

This dataset encapsulates 5 years of price data of the companies comprising the S&P 500, segmented into intervals of 30 trading days each. The data in each segment has been normalized using a method where values are divided by the most recent data point within the segment. Each row in the dataset represents a specific segment, providing a snapshot of the stock data available on a particular trading day. Rows are labeled to indicate when the stock had a minimum gain of 5% within the subsequent 10 trading days.

* `train.csv`: Of the 5 years, it contains the first 4 years of data.
* `test.csv`: Of the 5 years, it contains the final year of data.
* `latest.csv`: This file contains data from the most recent trading day for all stocks listed. While it lacks labels (since these pertain to future events), each row maintains the same feature vector structure as those in the `train` and `test` files. The rows commence with the stock ticker symbol, serving as a key tool to pinpoint stocks with promising prospects for good performance.

### CSV schema and ordering

- `train.csv` and `test.csv` have no header. Every column except the last is a
  numeric feature; the last column is the binary label (`0` or `1`).
- `latest.csv` has a header. Its first column is the ticker and its remaining
  columns must match the training features in number and order.
- Labeled rows must be sorted chronologically. If nearby windows share
  observations, set `EMBARGO_ROWS` so training and validation cannot overlap.
  Retain ticker/date metadata upstream so leakage can be audited.
- Reserve `test.csv` for evaluation after model and threshold selection.
  Repeatedly choosing models from test results leaks information.

The examples print precision, recall, F1, balanced accuracy, coverage, and a
Fisher exact-test p-value. For a trading study, provide realized returns to
`evaluation.backtest_top_scores` and choose realistic transaction costs.

### Getting new data

The example data is static and does not contain current stock price values.
Recent data customizable with different trading strategies and feature engineering options can be [downloaded for free at D.AT](https://d.at/ref/github-python-examples).

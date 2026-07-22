# Stock Prediction with Neural Networks and Machine Learning (Python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Stock prediction neural network and machine learning examples](https://repository-images.githubusercontent.com/669594930/1b661bb8-d5d8-40ad-9d94-c3084f3df2fc)

## Contents

- [Simple examples](#simple-examples)
- [Getting started](#getting-started)
- [Hyperparameter optimization](#hyperparameter-optimization)
- [About the example stock data](#about-the-example-stock-data)

## Simple Examples

This repository contains small, runnable machine learning (ML) and neural
network (NN) examples for stock classification. Each example focuses on the
essential parts of a technique: loading data, training a model, evaluating it,
and ranking the latest observations by predicted score.

> **Educational use only:** These examples are demonstrations, not investment
> advice or evidence of a profitable strategy. Classification scores do not
> account for slippage, liquidity, taxes, market impact, or regime changes.

### Machine learning examples

- Genetic algorithm
- Gradient boosting
- K-means clustering
- Logistic regression
- Random forest
- Support vector machine (SVM)

### Neural network architectures

- Feed-forward neural network (FFNN)
- Long short-term memory network (LSTM)
- Recurrent neural network (RNN)

The neural network examples use Keras, PyTorch, PyTorch Lightning, and
TensorFlow. Browse the implementations in
[`simple_examples`](simple_examples).

## Getting Started

Python 3.10, 3.11, or 3.12 is required.

1. Clone this repository and change to its directory.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Choose one installation command:

| What you want to run | Command |
| --- | --- |
| All examples | `pip install -r requirements.txt` |
| Classical ML only | `pip install -e .` |
| Classical ML, PyTorch, and Lightning | `pip install -e '.[torch]'` |
| Classical ML, TensorFlow, and Keras | `pip install -e '.[tensorflow]'` |

`requirements.txt` installs the project in editable mode with every supported
framework, so you do not need to run the other installation commands as well.

4. [Download the starter data](https://d.at/example-data/) and place the
   extracted `example_data` directory at the repository root:

```text
example_data/
├── latest.csv
├── test.csv
└── train.csv
```

5. Run an example from the repository root:

```bash
python -m simple_examples.machine_learning.logistic_regression
python -m simple_examples.neural_networks.pytorch_ffnn
```

## Hyperparameter Optimization

The neural network tuner supports configurable grid and random searches and
uses multiple processes to evaluate configurations.

1. Edit [`hyperparameter_tuning/config.py`](hyperparameter_tuning/config.py).
2. From the repository root, run:

```bash
python -m hyperparameter_tuning.hyper_main
```

The tuner uses the chronological tail of `train.csv` for model selection and
does not inspect `test.csv`. Configure `VALIDATION_FRACTION` and `EMBARGO_ROWS`
in `config.py`. Results are written to the `results` directory.

See the [hyperparameter tuning guide](hyperparameter_tuning/README.md) for the
search strategies and available settings.

## About the Example Stock Data

The examples use the [D.AT example dataset](https://d.at/example-data/).

The dataset contains five years of price data for S&P 500 companies, divided
into 30-trading-day windows. Values within each window are divided by the most
recent value in that window. A row is labeled positive when the stock gains at
least 5% during the following 10 trading days.

- `train.csv` contains the first four years of labeled data.
- `test.csv` contains the final year of labeled data.
- `latest.csv` contains one unlabeled row per stock for generating current
  rankings. Its first column contains the ticker symbol.

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

The example dataset is static and does not contain current prices. You can
[download recent customizable data from D.AT](https://d.at/ref/github-python-examples)
with additional strategy and feature-engineering options.

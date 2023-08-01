# Anti Money Laundering Machine Learning Project

## Overview

This repository contains the code and resources for an Anti Money Laundering (AML) Machine Learning Project. The goal of this project is to build a machine learning model that can detect and flag potential money laundering activities within financial transactions.

Link for the dataset - https://www.kaggle.com/code/x09072993/aml-detection/data

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data](#data)
5. [Preprocessing](#preprocessing)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Introduction

Money laundering is a serious financial crime that involves disguising illegal gains to make them appear legitimate. Financial institutions are obligated to implement robust AML measures to detect and prevent money laundering activities. In this project, we aim to develop a machine learning model that can help identify suspicious transactions and potential money laundering activities, thereby assisting financial institutions in maintaining compliance with AML regulations.

## Project Structure

The project has the following structure:

```
|-- data/
|   |-- raw_data.csv
|-- notebooks/
|   |-- data_exploration.ipynb
|   |-- data_preprocessing.ipynb
|   |-- model_training.ipynb
|   |-- model_evaluation.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- model.py
|-- models/
|   |-- trained_model.pkl
|-- requirements.txt
|-- README.md
```

1. `data/`: This directory contains the raw data file (`raw_data.csv`) used for training and evaluation.

2. `notebooks/`: This directory contains Jupyter notebooks for different stages of the project, including data exploration, data preprocessing, model training, and model evaluation.

3. `src/`: This directory contains the source code for data preprocessing (`preprocessing.py`) and model building (`model.py`).

4. `models/`: After training the model, it stores the serialized trained model (`trained_model.pkl`).

5. `requirements.txt`: This file contains the necessary dependencies and libraries required to run the project.

## Installation

To set up the project environment, you can use `pip` to install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Data

The raw data used for this project is stored in the `data/` directory. The data is in CSV format (`raw_data.csv`) and includes relevant features and labels to train the machine learning model.

## Preprocessing

The `notebooks/data_preprocessing.ipynb` notebook demonstrates the data preprocessing steps. These steps include data cleaning, feature engineering, handling missing values, and preparing the data for training the model.

## Model Training

The `notebooks/model_training.ipynb` notebook showcases the model training process. We use machine learning algorithms and techniques to train a model on the preprocessed data. The trained model is saved in the `models/` directory as `trained_model.pkl`.

## Model Evaluation

The `notebooks/model_evaluation.ipynb` notebook evaluates the performance of the trained model. Various metrics and visualizations are used to assess the model's effectiveness in detecting money laundering activities.

## Usage

To use the trained model for detecting money laundering activities, you can follow the steps outlined in the `notebooks/model_evaluation.ipynb` notebook. Additionally, you can integrate the trained model into your own applications or systems for real-time AML detection.

## Contributing

We welcome contributions to this project. If you find any issues or have ideas to improve the model or data processing, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---

Please ensure to replace the placeholders with actual data and instructions specific to your project. The README serves as a guide for users and contributors to understand your project's structure and usage.

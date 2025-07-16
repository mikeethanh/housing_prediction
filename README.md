# 🏠 House Price Prediction
This project focuses on predicting house prices using various tree-based machine learning algorithms. After experimenting with multiple models such as Random Forest, Gradient Boosting, and XGBoost, the best-performing model was selected and fine-tuned using hyperparameter optimization techniques.

The final model is deployed as an interactive web application using Streamlit, allowing users to input house features (e.g., size, number of bedrooms) and receive a predicted price in real-time.

## 🔍 Key Features:
Tree-based regression models for accurate predictions

Model comparison and selection through performance evaluation

Hyperparameter tuning for improved accuracy

Streamlit app for easy and interactive user experience

## Requirements

To run the code, you need to install the required packages. You can install the required packages using the following command:

```bash
conda create -n aio-mlops-w1 python=3.9.11 --y
conda activate aio-mlops-w1
pip install -r requirements.txt
```

## Streamlit

Before deploying the model, we need to train the model and save it. We will use the Random Forest Regressor model to predict the price of a house based on the number of bedrooms and bathrooms and the area of the house.

Follow the steps below to deploy the model using Streamlit:

### Step 1: Training a Model

The base code training is available in the notebook folder. You can run the code and train the model.

### Step 2: Saving the Model

Run the cell ***Save the model*** to save the model in the model folder.

```python
import pickle

with open("rf_regressor.pkl", "wb") as model_file:
    pickle.dump(rf_regressor, model_file)
```

### Step 3: Building a Web Application

First you need copy the model file (checkpoint) in notebook folder to this directory.

To start the streamlit application, run the following command in the terminal:

```bash
streamlit run streamlit_app.py
```


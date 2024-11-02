# Hotel Booking Prediction

This repository implements a machine learning model for detecting and predicting booking status using features from a hotel booking dataset. The project encompasses a complete data pipeline from data preprocessing, feature engineering, model training, evaluation, and deployment using a Flask app.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Project Workflow](#project-workflow)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Contributors](#contributors)

---

## Project Overview

This project is designed to classify booking status based on customer and booking attributes. Leveraging feature selection, handling imbalanced data, and implementing a RandomForest classifier, it provides an accurate and optimized model for prediction.

## Dataset

The dataset contains various features, such as:
- Customer and booking details
- Meal types
- Room types and other hotel-related information

You can access the dataset [here](https://docs.google.com/spreadsheets/d/1U0yXpYDtObVtduA4XIS-rDwk9U4rnhad7hXUabpVLfY/edit?usp=sharing).

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/project-name.git
   cd project-name
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Project Workflow

### 1. Data Loading and Initial Analysis
   - The data is loaded and basic information about the dataset is inspected to understand its structure, types, and potential preprocessing needs.
   - Exploratory data analysis (EDA) is performed to visualize distributions of key categorical features.

### 2. Data Preprocessing
   - **Cleaning**: White spaces are removed, and categorical features are encoded.
   - **Feature Engineering**: New features, like date components and total guests, are created.
   - **Encoding**: Categorical variables are one-hot encoded.

### 3. Handling Imbalance
   - `RandomOverSampler` from the `imbalanced-learn` library is used to balance the target classes.

### 4. Feature Selection and Splitting
   - Data is split into training and testing sets.
   - Feature selection techniques (e.g., `SelectKBest` and RFE with RandomForest) are applied to retain the most important features.

### 5. Model Training
   - A RandomForest model is trained on the selected features.
   - Feature importance is evaluated and retained for model optimization.

### 6. Model Evaluation
   - Model performance is assessed using metrics like confusion matrix and other relevant metrics.
   - Outlier detection and visualization are implemented to refine the data quality.

### 7. Model Export
   - The trained model is saved as `model.pkl` using `pickle` and `joblib` for deployment purposes.
   - The model will be loaded when the notebook is exported, and the `.pkl` file will be placed in the appropriate folder.

## Model Evaluation

Confusion matrix and classification reports are used to evaluate the model. Visualization techniques like boxplots and pie charts provide insights into the class distribution and outliers.

## Deployment

The project is deployed using a Flask application (`app.py`), which:
- Loads the saved model (`model.pkl`).
- Exposes an endpoint for predictions based on user inputs through a web form.

To run the application:
```bash
python app.py
```

Access the application locally at `http://127.0.0.1:5000`.

## Contributors

- [Malak Amgad](www.linkedin.com/in/malak-amgad-9a6892261) - AI Engineer, Developer

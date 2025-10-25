# Furina ML

A modern web application for machine learning model development with an intuitive user interface. This application provides a complete workflow for preprocessing data, training models, and evaluating results.

## Features

### Data Preprocessing
- Upload and preview dataset files
- Handle missing values
- Feature scaling and normalization
- Feature selection capabilities
- Data visualization tools

### Model Training
- Multiple model options:
  - Scikit-learn Logistic Regression
  - Scikit-learn Support Vector Machine
  - TensorFlow Neural Network
  - Scikit-learn Random Forest
- Customizable train/test split ratio
- Adjustable hyperparameters
- Random seed setting for reproducibility
- Target column and feature selection

### Model Evaluation
- Comprehensive metrics:
  - Accuracy
  - Precision
  - Recall
  - F2 Score
- Detailed confusion matrix
- Interactive prediction testing
- Model and report downloads

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/thirza258/Furina-ML.git
   cd Furina-ML
   ```

2. Start the application:
   ```bash
   docker compose up --build
   ```

3. Access the application:
    ```
    backend : localhost:8000
    frontend : localhost
    ```
# Bank Marketing Prediction Analysis  
  
This repository contains a machine learning project aimed at analyzing and predicting outcomes from a bank marketing dataset. The project utilizes logistic regression and XGBoost models to predict whether a client will subscribe to a term deposit.  
  
## Overview  
  
The program performs the following operations:  
  
- Fetches the Bank Marketing Dataset from UCI Machine Learning Repository.  
- Splits the data into training and testing sets.  
- Preprocesses the data, including handling missing values, scaling features, and data transformations.
- Trains Logistic Regression and XGBoost models on the training set.  
- Makes predictions and evaluates the models on the testing set using accuracy scores and classification reports.  
- Generates confusion matrices and feature importance rankings.  
  
## Prerequisites  
  
- Python 3.x  
- Pandas  
- Matplotlib  
- Scikit-learn  
- XGBoost (if using the XGBoost model)  
  
Ensure `matplotlib` is configured to use 'Qt5Agg' for the backend as shown in the code.  
  
## Installation  
  
To run this program, you need to install the required libraries. You can install them using `pip`:  
  
```bash  
pip install pandas matplotlib scikit-learn xgboost  
```

## Usage  
  
1. Import necessary modules and fetch the dataset using the provided custom function `fetch_ucirepo`.  
2. Perform data preprocessing and transformations.  
3. Train the logistic regression and XGBoost models using the `model_functions` module.  
4. Evaluate the models and interpret the results through confusion plots and classification reports.  
  
## Dataset  
  
The dataset is fetched from the UCI Machine Learning Repository using a custom function `fetch_ucirepo`. It includes customer data from a bank's marketing campaign.  
  
## Model Training  
  
Two models are trained in this program:  
  
- Logistic Regression: Scaled features are used to train the logistic regression model.  
- XGBoost: Non-scaled features are used to train the XGBoost model.  
  
## Model Evaluation  
  
The performance of both models is evaluated using classification reports that provide precision, recall, f1-score, and accuracy. Confusion matrices are generated to visualize the true positives, true negatives, false positives, and false negatives.  
  
## Feature Importance  
  
Feature importance scores are extracted from the trained XGBoost model to understand the influence of each feature on the model's predictions.  
  
## Contributing  
  
Feel free to fork this repository, make changes, and submit pull requests. Please open an issue to discuss any significant work to coordinate and avoid duplication of effort.  
  
## License  
  
This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for details.  
  
## Acknowledgments  
  
This project utilizes data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).  
  
## Contact  
  
For any queries or feedback, please open an issue in the repository, and I will get back to you.  

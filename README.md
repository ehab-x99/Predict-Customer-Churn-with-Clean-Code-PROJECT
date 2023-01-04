# Predict Customer Churn 

## Project Description
This project aims to predict customer churn using machine learning algorithms. The goal is to build models that can accurately predict whether a customer will churn or not, based on various features such as gender, education level, and income.

The code is written in Python and follows PEP 8 style guidelines to ensure readability and maintainability. The data is read from a CSV file and cleaned and preprocessed using pandas. The models are trained using scikit-learn and the results are visualized using matplotlib.


# Files description
1. ***churn_notebook.ipynb***:
-This is The notebook which the project runs and visualize data inside of it.

2. ***churn_library.py***: 
- This library Contains the functions to find customers who are likely to churn.

3. ***churn_script_logging_and_tests.py***:
- Contain unit tests for the *churn_library.py* functions. 
- Log any errors and INFO messages.
- Saves Logs in the Log file

## Running Files instructions
ivate churn_predict
```
1. Install packages:
```bash
conda pip install --file requirements.txt
```
2. Run churn prediction:
```bash
ipython churn_library.py
```
3. Test churn prediction:
```bash
ipython churn_script_logging_and_tests.py
```





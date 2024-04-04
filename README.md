# AIPM_Final

## Project Background

RetailGenius is a rapidly growing e-commerce company with a diverse range of products, sellers, and customers. They
operate globally and handle a substantial volume of data.

Retail Genius launched a strategic program to derive value from its huge amount of Data. To achieve this goal, they
started a first Al project to predict customers churn.

## Brief Intro

This project allow users:
1. Check the performance of model in use(shap)
2. Check the life circle of model in use(mlflow)
3. Check the documents of scripts(sphinx)
4. Predict the churn of customer(RandomForest model)

## Install

`pip install -r 'requirements.txt'`

**Additional Package**

sphinx: `pip install sphinx sphinx_rtd_theme`

cookiecutter: `pip install cookiecutter`

## Run

Mlflow: `mlflow server`
Sphinx: 
```
cd docs
make clean
make html
```





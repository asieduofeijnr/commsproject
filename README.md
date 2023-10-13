# Height Prediction Using Multiple Linear Regression

This is a simple Streamlit application for predicting a person's height based on their shoe size and gender using Multiple Linear Regression (MLR). It utilizes various Python libraries and offers an interactive way to estimate height.

## Overview

The application allows you to input your gender and shoe size and will predict your height using a trained MLR model. Before getting into predictions, let's dive into the details of this project.

## Contributors

This project was designed by a team of developers:
- Andrea
- Felix
- Tyler
- Solomon

## Model

### Data

The dataset used for this application is loaded from a CSV file named 'wo_men.csv'. It contains three columns: 'sex', 'shoe_size', and 'height'. The 'sex' column represents gender, 'shoe_size' represents the shoe size in EU size, and 'height' represents the actual height.

### Model Details

The MLR model is constructed with 'shoe_size' and 'sex' as predictors and 'height' as the target variable. The application also checks for multicollinearity using the Variance Inflation Factor (VIF) to ensure the model's reliability.

```python
model = smf.ols('height ~ shoe_size + sex', df).fit()
model.summary()
```

The model is fitted, and a summary of the model's statistics is displayed in the application.

### Residual Analysis
The application provides a graphical representation of residuals to assess the model's performance. It plots 'Fitted Values vs. Residuals' to help you understand the model's performance better.

### Outliers
The application identifies and highlights any unusual data points, which are considered as potential outliers. These outliers are then removed from the dataset, and the model is retrained to obtain a more robust model.

The updated model is displayed in the application, along with the revised 'Fitted Values vs. Residuals' plot.

### Predicting Height
After preprocessing the data and updating the model, you can make predictions for your height. Simply select your gender ('Man' or 'Woman') and input your shoe size in EU size. The application uses the trained model to predict your height and displays the result.

Important Libraries
This application uses various Python libraries for data analysis and visualization:


# Presentation Overview

## Introduction by Andrea
Andrea will kick off the presentation by introducing the team and setting the stage for the presentation. She will also provide a brief summary of the topics covered in previous presentations and explain how today's discussion builds upon that knowledge. Andrea will emphasize the practicality of linear regression as a modeling method, which is the central theme of our presentation. She will delve into the predictive and inferential aspects of linear regression that make it intriguing.

## Advantages of Linear Regression by Tyler
Tyler will follow up by discussing the advantages of linear regression and what makes it a unique method for modeling data. He will briefly touch on the motivation behind our example but primarily focus on explaining why the results are interesting. Tyler will emphasize the significance of making inferences and highlight the explainability of linear regression, a feature that sets it apart from many popular machine learning models.

## Model Explanation by Felix
Felix will provide a detailed explanation of the model being used, with a focus on accuracy and practical applications. He will clarify what R-squared is and how it is used to evaluate our model. Felix will also discuss the identification of statistically significant predictors and how the model's outputs can be interpreted in more detail. Additionally, he will touch on some of the drawbacks of linear regression and suggest ways to improve model performance.

## Interactive Example by Solomon
Solomon will conclude the presentation with an interactive example of linear regression. He will engage the audience by collecting their shoe sizes and attempting to predict their heights using the model. Solomon will explain how to interpret the model's prediction output, including the difference between the actual prediction and the prediction interval. The goal of this section is to showcase linear regression in action and highlight its prediction capabilities, even in the face of the model's simplicity.

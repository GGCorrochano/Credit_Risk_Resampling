# Overview

Credit Risk Resampling uses historical lending activity data, from a peer-to-peer lending services company, to build a model that can identify the creditworthiness of borrowers.

# Usage

Credit Risk Resampling uses a logistic regression model to compare two versions of the dataset. First, we use the original dataset. Second, we resample the data by using the RandomOverSampler module from the imbalanced-learn library in order to get the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

Loans identified by indicate a healthy loan. Loans identified by 1 indicate a high risk or default. 

# Results

**Machine Learning Model 1:

  * Description of Model 1
    ***Fit the model using training data 
       logistic_regression_model.fit(training_features, training_targets)
    ***Make a prediction using the testing data
      testing_prediction = logistic_regression_model.predict(testing_features)
  * Accuracy
    ***Print the balanced_accuracy score of the model
       balanced_accuracy_score(testing_targets, testing_prediction)
       0.9436265644476789
  * Precision
    ***Generate a confusion matrix for the model
       confusion_matrix(testing_targets, testing_prediction)
       array([[18668,    87],
              [   68,   561]])
  * Recall scores
    ***Print the classification report for the model
       print(classification_report_imbalanced(testing_targets, testing_prediction))
                              pre       rec       spe        f1       geo       iba       sup

                     0       1.00      1.00      0.89      1.00      0.94      0.90     18755
                     1       0.87      0.89      1.00      0.88      0.94      0.88       629

           avg / total       0.99      0.99      0.90      0.99      0.94      0.90     19384

**Machine Learning Model 2:

  * Description of Model 2 
    ***Instantiate the Logistic Regression model. Assign a random_state parameter of 1 to the model
       model_2 = LogisticRegression(random_state=1)
       Fit the model using the resampled training data
       model_2.fit(X_resampled, y_resampled)
       Make a prediction using the testing data
       y_predict = model_2.predict(X_resampled)
  * Accuracy
    ***Print the balanced_accuracy score of the model 
       balanced_accuracy_score(testing_targets, testing_prediction)
       0.9936781215845847
  * Precision
    ***Generate a confusion matrix for the model
       confusion_matrix(testing_targets, testing_prediction)
       array([[18649,   116],
              [    4,   615]])
  * Recall scores
    ***Print the classification report for the model
       print(classification_report_imbalanced(testing_targets, testing_prediction))
                         pre       rec       spe        f1       geo       iba       sup

                0       1.00      0.99      0.99      1.00      0.99      0.99     18765
                1       0.84      0.99      0.99      0.91      0.99      0.99       619

      avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384

## Summary

Our analysis returns an acceptable accuracy from both models (Model 1 & Model 2). 

* Which one seems to perform best? How do you know it performs best?
  At 99% accuracy, Model 2 outperforms Model 1. (Model 1 accuracy = 94%).
  
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
  Yes. It is more beneficial for lenders to identify & make provisions for borrowers that are high risk & likely to default, 
  than to focus on borrowers with good standing loans.

# Contributing

Pull requests are welcome. Please open an issue to discuss before executing any changes.
Please make sure to update tests as needed.

# License
BSD-2-Clause https://opensource.org/licenses/BSD-2-Clause

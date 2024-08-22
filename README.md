# KNN_Impl

Implementing KNN from scratch in Python, then using it to predict diabetes in patients using Numpy, Pandas, Matplotlib, and Scikit-learn. Testing for various k values, cross-validation scores and across different PCA number of components.

## Running the Code

To run the code, you will need to have Python installed on your system. You can download Python from the official website [here](https://www.python.org/downloads/). The code is written in Python 3.11.0, so it is recommended to use this version of Python to run the code.

To install the required libraries, you can use the following command:

```bash
pip install -r requirements.txt
```

To run the code, you can use the following command:

```bash
python main.py
```

The following commands will run the code and display results in the console, as well as plot the graphs for the results found.

## About the KNN Algorithm

The KNN algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. It is a non-parametric method that makes no assumptions about the underlying data distribution. The algorithm works by finding the k-nearest neighbors of a given data point and then making a prediction based on the majority class (for classification) or the average value (for regression) of those neighbors.

- fit(x_train, y_train, method): This method is used to train the model on the training data. It takes the training features (X_train) and labels (y_train) as input and stores them in the model. The method parameter is used to specify the distance metric for the algorithm to be used for finding the nearest neighbors, the default value is 'euclidean'.
- predict(x_test): This method is used to make predictions on the test data. It takes the test features (x_test) as input and returns the predicted labels for those features.

The algorithm uses the brute force method to find the nearest neighbors, which involves calculating the distance between the test data point and all the training data points, and then selecting the k-nearest neighbors based on the distance metric specified.

## About the Dataset

The dataset used is the [Diabetes Dataset For Beginners](https://www.kaggle.com/code/melikedilekci/diabetes-dataset-for-beginners) from Kaggle, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The selected dataset contains 9 columns and 768 entries. The columns are as follows:
1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. DiabetesPedigreeFunction: Diabetes pedigree function
8. Age: Age (years)
9. Outcome: Class variable (0 or 1), where 1 is positive for diabetes and 0 is negative for diabetes

The dataset is split into 75% training and 25% testing data. The KNN algorithm is implemented to predict the outcome of the testing data, having its base k value set to 5, using, mainly, the Euclidean distance as the distance metric and brute force as the algorithm to find the nearest neighbors.

## Testing the Algorithm

### Pre Processing

The model is tested before and after pre-processing the dataset.

The dataset is pre-processed by replacing removing the columns with low correlation with the outcome, as seen in the correlation matrix. The columns removed are 'Blood Pressure' and 'SkinThickness'.

Furthermore, the dataset has its outliers removed by calculation the quantiles of the columns and removing the rows that are outside the 1.5 * IQR range. Then, the values are normalized using the StandardScaler from the Scikit-learn library.

### Testing for Different k Values, Cross-Validation Scores, and PCA Components

Firstly, the model is tested for different k values, ranging from 1 to 100 (2 by 2), to find the optimal k value for the model. The accuracy of the model is calculated for each k value, with the cross-validation being a fixed value of 10.

Secondly, the model is tested for different cross-validation scores, ranging from 3 to 16, AND different k values, ranging from 1 to 100 (2-step), to find the optimal combination of cross-validation score and k value for the model.

Then, the model is tested for different PCA components, ranging from 1 to 8, with a fixed k value of 12 and cross-validation score of 10, to find the optimal number of PCA components for the model. Its visible that the model performs best without the use of PCA components.

However, further PCA value testing is still maintained for the sake of the experiment. The last graph shows the model's accuracy for different PCA components, ranging from 1 to 5, with a k value ranging from 9 to 56 (2-step) and a cross-validation score ranging from 3 to 16.

The model is executed for the last time with the optimal k value, cross-validation score, and PCA components found in the previous tests for the best possible model.

## Results

The results are analyzer by observing the model's accuracy and recall, as it is important to have a high recall value for the model to be able to predict diabetes in patients, as fake negatives are more dangerous than fake positives.

The model's accuracy is calculated for each test and plotted in a graph to visualize the results. The model's accuracy is calculated by comparing the predicted values with the actual values of the testing data.

For the model without the pre-processing, the accuracy is 68.18% and the recall was 37.00%, while the model with the pre-processing has an accuracy of 68.64% and recall of 48.00%. The model with the pre-processing has a slightly better accuracy than the model without the pre-processing, moreover, the model gets better results with better k and cross-validation values.

The best model with the best results and usage of PCA components has an accuracy of 77.00% and recall of 50.00%, with a k value of 31, a cross-validation score of 8, and 2 PCA components.

The model with the best results overall has a k value of 17, a cross-validation score of 6, and no PCA components, with an accuracy of 78.34% and recall of 52.00%.

## Conclusion

As the KNN algorithm is known to have worse results with higher dimensions, the usage of KNN for predicting diabetes in patients was a bad approach for this dataset, as the model's accuracy and, specially, recall were low. The model could be improved by using other algorithms, such as Logistic Regression, Random Forest, or Neural Networks, to predict diabetes in patients.

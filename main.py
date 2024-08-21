import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
import seaborn as sns
from KNN import KNN
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# Visualizing the dataset

diabetes = pd.read_csv('diabetes.csv')
# print all columns in 1 line
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("The first 5 rows of the dataset are:")
print(diabetes.head(5))
diabetes.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.suptitle("Dataset columns")
plt.show()

X, y = diabetes.iloc[:, :-1].values, diabetes.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNN()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test) # number of correct predictions divided by total number of predictions

print("\n\nAfter training the model with the dataset without preprocessing the data, the results are:")
print(f"{accuracy * 100:.2f}% is the accuracy of the model without preprocessing the data\n")
print(classification_report(y_test, predictions))
print("\n")

# Correlation matrix for the dataset without preprocessing
x_clean_df = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                      'DiabetesPedigreeFunction', 'Age'])
x_clean_df['Outcome'] = y
corr = x_clean_df.corr()
sns.heatmap(corr, annot=True)
plt.suptitle("Correlation matrix for the dataset without preprocessing")
plt.show()


#################################################


# Removing columns with low correlation to the outcome, based on the correlation matrix
# Convert x_clean to a DataFrame
x_df = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                      'DiabetesPedigreeFunction', 'Age'])

# Select only the desired columns in the non-outcome columns, based on the correlation matrix results
x_clean_df = x_clean_df[['Pregnancies', 'Glucose', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']]

# Plot the selected columns
x_clean_df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.suptitle("Selected columns for model")
plt.show()


#################################################


# Removing outliers from all columns except the last one
indices_to_keep = np.ones(len(x_clean_df), dtype=bool)
for column in x_clean_df.columns:
    q1 = x_clean_df[column].quantile(0.25)
    q3 = x_clean_df[column].quantile(0.75)
    iqr = q3 - q1
    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr

    column_indices_to_keep = (x_clean_df[column] >= inferior) & (x_clean_df[column] <= superior)
    indices_to_keep &= column_indices_to_keep
x_clean_df = x_clean_df[indices_to_keep].reset_index(drop=True)
y = y[indices_to_keep]

# Plot the cleaned data
x_clean_df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.suptitle("Cleaned data, without outliers")
plt.show()


#################################################


# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_clean = scaler.fit_transform(x_clean_df)

# Use the original column names for the normalized DataFrame
x_clean_plot = pd.DataFrame(x_clean, columns=x_clean_df.columns)
x_clean_plot.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
plt.suptitle("Normalized data")
plt.show()


#################################################


knn = KNN()
X_train, X_test, y_train, y_test = train_test_split(x_clean, y, test_size=0.25)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
knn_accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"{knn_accuracy * 100:.2f}% is the accuracy of the model after preprocessing the data\n")
print(classification_report(y_test, predictions))


#################################################


# Using cross_validation to get the best k value
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
k = list(range(1, 100))
scores = []
print("\nCalculating for [1, 100] k values...")
for i in k:
    knn = KNN(i)
    predictions = cross_val_predict(knn, x_clean, y, cv=10)
    scores.append(accuracy_score(y, predictions))

plt.plot(k, scores)
plt.suptitle("Accuracy for different k values")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

best_k = k[scores.index(max(scores))]
print(f"The best k value is {best_k} for a 10-fold cross validation with a accuracy of {max(scores)*100}%")


#################################################


k = list(range(9, 55))
cv = list(range(3, 15))
best_scores = []
cv_for_best_scores = []
best_result = [0, 0]
print("\nCalculating best cv [3, 15] value for each k [9, 55] value...")
for i in k:
    scores = []
    for j in cv:
        knn = KNN(i)
        predictions = cross_val_predict(knn, x_clean, y, cv=j)
        scores.append(accuracy_score(y, predictions))
    best_scores.append(max(scores))
    cv_for_best_scores.append(cv[scores.index(max(scores))])
    if max(scores) > best_result[0]:
        best_result = [max(scores), i]
print(f"\nThe best k value is {best_result[1]} with a cv value of {cv_for_best_scores[k.index(best_result[1])]}, "
      f"with an accuracy of {best_result[0]*100}%")

plt.plot(k, best_scores)
plt.suptitle("Accuracy for different k values with best cv value")
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()


#################################################


# Using PCA to reduce the number of features and test results
from sklearn.decomposition import PCA
pca_range = list(range(1, 4))
k_value = 12
scores = []
print("\nCalculating for [1, 3] PCA components...")
for i in pca_range:
    pca = PCA(n_components=i)
    x_pca = pca.fit_transform(x_clean)
    knn = KNN(k_value)
    predictions = cross_val_predict(knn, x_pca, y, cv=10)
    scores.append(accuracy_score(y, predictions))

plt.plot(pca_range, scores)
plt.suptitle("Accuracy for different PCA components, with k = 12 and no cross validation")
plt.xlabel('PCA components')
plt.ylabel('Accuracy')
plt.show()

# Since the accuracy with the PCA components is lower than the accuracy without PCA, we will not use PCA for the
# final test, only the best k and cv values found


#################################################

# Using the best k value and cv value for last test without PCA

knn = KNN(best_result[1])
predictions = cross_val_predict(knn, x_clean, y, cv=cv_for_best_scores[k.index(best_result[1])])
accuracy = accuracy_score(y, predictions)
print(f"{accuracy * 100:.2f}% is the accuracy of the model after preprocessing the data, with the usage of the best "
      f"k, best cv and no PCA usage")
print(classification_report(y, predictions))

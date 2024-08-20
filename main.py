import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Visualizing the data
diabetes = pd.read_csv('diabetes.csv')
# print all columns in 1 line
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(diabetes.head(5))
diabetes.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

X, y = diabetes.iloc[:, :-1].values, diabetes.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNN()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test) # number of correct predictions divided by total number of predictions

print(f"{accuracy * 100:.2f}% is the accuracy of the model without preprocessing the data")
from sklearn.metrics import classification_report
print(classification_report(y, predictions))


# Confusion matrix for dataset without preprocessing
import seaborn as sns
x_clean_df = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                      'DiabetesPedigreeFunction', 'Age'])
x_clean_df['Outcome'] = y
corr = x_clean_df.corr()
sns.heatmap(corr, annot=True)
plt.show()


#################################################


# Removing columns with low correlation to the outcome
# Convert x_clean to a DataFrame
x_df = pd.DataFrame(X, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                      'DiabetesPedigreeFunction', 'Age'])

# Select only the desired columns in the non-outcome columns, based on the correlation matrix results
x_clean_df = x_clean_df[['Pregnancies', 'Glucose', 'BMI', 'Age']]

# Plot the selected columns
x_clean_df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
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
x_clean_df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

#################################################


# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_clean = scaler.fit_transform(x_clean_df)

# Use the original column names for the normalized DataFrame
x_clean_plot = pd.DataFrame(x_clean, columns=x_clean_df.columns)
x_clean_plot.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# After using PCA to reduce the number of features, the results were worse
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# x_clean = pca.fit_transform(x_clean)


#################################################


knn = KNN()
X_train, X_test, y_train, y_test = train_test_split(x_clean, y, test_size=0.25)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
knn_accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"{knn_accuracy * 100:.2f}% is the accuracy of the model after preprocessing the data")
from sklearn.metrics import classification_report
print(classification_report(y, predictions))


#################################################


# Using cross_validation to get the best k value
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
k = list(range(1, 100))
scores = []
print("\nCalculating k values...")
for i in k:
    knn = KNN(i)
    predictions = cross_val_predict(knn, x_clean, y, cv=10)
    scores.append(accuracy_score(y, predictions))

plt.plot(k, scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

best_k = k[scores.index(max(scores))]
print(f"The best k value is {best_k} for a 10-fold cross validation")


#################################################


k = list(range(15, 70))
cv = list(range(4, 15))
best_scores = []
cv_for_best_scores = []
best_result = [0, 0]
print("\nCalculating best cv value for each k value...")
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
    print(f"Best cv value for k = {i} is {cv[scores.index(max(scores))]}")
print(f"\nThe best k value is {best_result[1]} with a cv value of {cv_for_best_scores[k.index(best_result[1])]}")

plt.plot(k, best_scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

#################################################

# Using the best k value and cv value

knn = KNN(best_result[1])
predictions = cross_val_predict(knn, x_clean, y, cv=cv_for_best_scores[k.index(best_result[1])])
accuracy = accuracy_score(y, predictions)

from sklearn.metrics import classification_report
print(classification_report(y, predictions))

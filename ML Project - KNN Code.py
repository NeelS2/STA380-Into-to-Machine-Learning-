import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree


df = pd.read_csv("WineQT.csv")
print(df.iloc[0]) # Slide 5 on the presentation - just showing a sample of the first row of data
y = df['quality']
# # X = df[['alcohol', 'sulphates', 'pH', 'volatile acidity', 'citric acid']] # These were 'optimal', didn't feel like
# it was worth the slight loss in accuracy score for a less complex model
X = df.drop(columns=['quality', 'Id']) # Better

# Scale the data
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1000)

knn = KNeighborsClassifier()
# GridSearchCV to run cross-validation, gives optimal k and parameters, took these into account
k_range = list(range(1, 101))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
grid_search = grid.fit(X_train, y_train)
print(grid_search.best_params_)

# Loop to get the data to create the graph shown in slide 8, accuracy scores for training and testing vs number of k's
ks = np.arange(1, 100)
accuracy_train = []
accuracy_test = []
for k in ks:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    accuracy_train.append(model_knn.score(X_train, y_train))
    accuracy_test.append(model_knn.score(X_test, y_test))

# Creating the graph in slide 8
plt.plot(ks, accuracy_train)
plt.plot(ks, accuracy_test)
plt.legend(['Training', 'Test'])
plt.xlabel('Number of Neighbors, k')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for the Training, and Test Data vs. Number of Neighbors Picked')
plt.show()

# Print the highest scores for train and test
highest_k_train = accuracy_train.index(max(accuracy_train))
highest_k_test = accuracy_test.index(max(accuracy_test))
print(highest_k_test)
print(max(accuracy_test))

# Pick 17 as the k for the final model and create confusion amtrix (shown in slide 9)
model_knn_final = KNeighborsClassifier(n_neighbors=17)
model_knn_final.fit(X_train, y_train)
y_pred = model_knn_final.predict(X_test)
plot_confusion_matrix(model_knn_final, X_test, y_test)
plt.grid(False)
plt.title('Confusion Matrix for Test Data on Wine Quality')
plt.ylabel('Actual label')
plt.show()

final_score = model_knn_final.score(X_test, y_test)
print("The accuracy score of the model was " + str(model_knn_final.score(X_test, y_test)))

# Print the final score and the dataset with the predicited quality (shown in slide in 10)
X_test['Quality'] = y_test
X_test['Predicted Quality'] = y_pred
print(X_test)
print("The accuracy score of the model was " + str(final_score))





# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:50:04 2021

@author: noahs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris_Data.csv')
X = dataset.iloc[:,:-1].to_numpy() 
y = dataset.iloc[:, -1].to_numpy()

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scale = sc.fit_transform(X)
#y_scale = sc.fit_transform(y.reshape(len(y), 1)).flatten()

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_scale, y)

# Predicting the Test set results
y_pred = classifier.predict(X_scale)

# Showing the Confusion Matrix and Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y, y_pred))
print(accuracy_score(y, y_pred))

#linear - .8133
#polynomial - .7466
#rbf - .82
#sigmoid - .7733
#RBF had most accurate score


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_scale, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                               stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.
             predict(np.array([X1.flatten(), X2.flatten()]).T).
             reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','blue')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color = ListedColormap(('red', 'green','blue'))(i),
                edgecolors = 'black', label = j)
plt.title('Kernel SVM')
plt.xlabel('Sepal Length (Scaled)')
plt.ylabel('Sepal Width (Scaled)')
plt.legend()
plt.show()





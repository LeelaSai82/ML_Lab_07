#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdata (1).xlsx")
df


# In[2]:


import numpy as np
import pandas as pd
from sklearn import svm

df = pd.read_excel("embeddingsdata (1).xlsx")
# Assuming you already have a DataFrame 'df' with a column 'Label' containing 0 or 1
binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

# Create an SVM classifier
clf = svm.SVC()

# Fit the classifier to the data
clf.fit(X, y)

# Get the support vectors
support_vectors = clf.support_vectors_

# Print the support vectors
print("Support Vectors:")
print(support_vectors)


# In[12]:


import numpy as np
from sklearn import svm

# Create an SVM classifier
clf = svm.SVC()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Test the accuracy of the SVM on the test set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)


# In[17]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming you've already trained your SVC classifier (clf) and have a test dataset X_test
# Make predictions on the test dataset
predicted_labels = clf.predict(X_test)

# Now, you can compare the predicted labels to the true labels in your test set to calculate accuracy
true_labels = df.loc[X_test.index, 'Label'][df['Label'].isin([0, 1])].values

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[19]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split your data into training and testing sets
binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Define a list of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Create an SVM classifier with the specified kernel
    clf = SVC(kernel=kernel, random_state=42)
    
    # Train the SVM model on the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the test data
    predicted_labels = clf.predict(X_test)
    
    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"Accuracy with {kernel} kernel: {accuracy * 100:.2f}%")


# In[ ]:





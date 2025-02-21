from malconv import Malconv
from preprocess import preprocess
import utils
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data.csv", header=None)
filenames, label = df[0].values, df[1].values

#Find the unique classes
unique_classes = np.unique(label)
#Count the number of unique classes
number_classes = len(unique_classes)

model = Malconv(num_classes=number_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

seq, len_list = preprocess(filenames, 200000)
x_train, x_test, y_train, y_test = sk_train_test_split(seq, label)

history = model.fit(x_train, y_train)

pred = model.predict(x_test)
pred_classes = np.argmax(pred, axis=1) # Get the index of the highest probability

cm = sk_confusion_matrix(y_test, pred_classes)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))  # Adjust the size of the plot as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') # creates a "heat map" which is a color coded visual representation of the matrix.
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

#plt.show()
plt.savefig('confusion_matrix.svg', format='svg')
plt.close()
print("Confusion matrix saved as confusion_matrix.svg")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Step 1: Load and preprocess the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
names = ['letter', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
dataset = pd.read_csv(url, names=names)

# Splitting into features and target
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

# Normalize features (optional but can help model convergence)
X = X / 15.0  # assuming 15 is the maximum value for any feature

# Encoding target labels
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=42)

# Step 2: Define the model architecture
model = Sequential()
model.add(Dense(64, input_dim=16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(26, activation='softmax'))  # 26 classes for letters A-Z

# Step 3: Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)

# Step 5: Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f%%' % (accuracy * 100))

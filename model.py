import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the data
heart_data = pd.read_csv('data.csv')

# Selecting the 13 features for the model
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the model and scaler
joblib.dump(model, 'heart_disease_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Evaluate the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print(f'Accuracy on Training Data: {training_data_accuracy}')
print(f'Accuracy on Test Data: {test_data_accuracy}')

# Test prediction
input_data = (45, 0, 1, 112, 160, 0, 1, 138, 0, 0, 1, 0, 2)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
input_data_as_numpy_array = scaler.transform(input_data_as_numpy_array)  # Transform with the scaler

prediction = model.predict(input_data_as_numpy_array)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')

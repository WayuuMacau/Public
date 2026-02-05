import pandas as pd
from sklearn.naive_bayes import GaussianNB
from pandas import DataFrame

# Load the CSV file into a DataFrame
Data = pd.read_csv('Nasdaq.csv')
df = DataFrame(Data)

# Exclude the first (date) column and the last (dependent) column, except for the last row
X = df.iloc[1:-1, 1:-1]

# Only select the last (dependent) column, except for the last row
y = df.iloc[1:-1, -1].astype('int')  # Ensure the target variable is integer for classification

# Get the data for prediction (excluding the first (date) column and the last (dependent) column) from the last row
new_X = df.tail(1).iloc[:, 1:-1].astype(float)

# Create and fit the Naive Bayes classifier
model = GaussianNB()
model.fit(X, y)

# Predict on the known data
y_pred = model.predict(X)
print("Predictions for training data:", y_pred)

# Predict a new, single instance
y_new_pred = model.predict(new_X)
print("Prediction for new data:", y_new_pred)




import csv

# Define the output file name
output_file = 'output.csv'

# Read the existing data
with open(output_file, 'r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# Append '2' to the last row in the data
if data:
    data[-1].extend(y_new_pred)

# Now write the updated data back to the file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

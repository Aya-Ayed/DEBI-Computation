# -*- coding: utf-8 -*-
"""Copy of Spaceship Titanic

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XmF87Ot-VaJP5jSpM71Wn85cmo_dC40q
"""

from google.colab import drive
drive.mount('/content/drive/')

!unzip '/content/drive/MyDrive/spaceship-titanic.zip' -d spaceship-titanic

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

train_data=pd.read_csv("/content/spaceship-titanic/train.csv")
test_data=pd.read_csv("/content/spaceship-titanic/test.csv")

train_data.head()

train_data.info()

test_data.head()

train_data.dropna(inplace=True)
test_data.fillna(0,inplace=True)

train_data.drop(columns=['PassengerId', 'Name', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP'], inplace=True)

train_data.head()

x = train_data.drop("Transported",axis=1).astype(int)
y = train_data["Transported"].astype(int)

x_train,x_val, y_train, y_val= train_test_split(x, y, test_size=0.2, random_state=42)

LR = LogisticRegression();
LR.fit(x_train, y_train);

x_train

y_val_pred=LR.predict(x_val)

val_acc=accuracy_score(y_val, y_val_pred)
val_acc

# Ensure that the test data is processed similarly to the training data
test_data_encoded = pd.get_dummies(test_data, columns=['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP'])
test_data_encoded = test_data_encoded.drop(['Name'], axis=1)
test_data_encoded = test_data_encoded.fillna(0)
test_data_encoded = test_data_encoded[train_data.columns.drop("Transported")]

# Make predictions on the test set
test_predictions = LR.predict(test_data_encoded)

# Create a DataFrame with 'PassengerId' and 'Transported' columns
submission_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': test_predictions.astype(int)})
print(submission_df.head())
# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
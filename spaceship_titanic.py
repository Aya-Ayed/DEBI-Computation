import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("spaceship-titanic/train.csv")
test_data = pd.read_csv("spaceship-titanic/test.csv")

train_data.head()

train_data.info()

test_data.head()

train_data.dropna(inplace=True)
test_data.fillna(0, inplace=True)

train_data.drop(
    columns=[
        "PassengerId",
        "Name",
        "HomePlanet",
        "CryoSleep",
        "Cabin",
        "Destination",
        "VIP",
    ],
    inplace=True,
)

train_data.head()

x = train_data.drop("Transported", axis=1).astype(int)
y = train_data["Transported"].astype(int)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

LR = LogisticRegression()
LR.fit(x_train, y_train)

pickle.dump(LR, open("titanic.pkl", "wb"))

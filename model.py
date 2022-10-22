import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import pickle

df = pd.read_csv('mushrooms.csv')

# saving Ordinal Encoder object for later use to transform data receive from user
oe = OrdinalEncoder()
X = df.drop(['class', 'veil-type'], axis=1)
y = df['class']
y = np.array(y).reshape(-1, 1)

X = oe.fit_transform(X)
with open(r"oe.pickle", "wb") as output_file:
    pickle.dump(oe, output_file)

y = oe.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=102)

# define models and parameters
model = BaggingClassifier(n_estimators=10)
model.fit(X_train, y_train)

with open(r"my_model.pickle", "wb") as obj:
    pickle.dump(model, obj)

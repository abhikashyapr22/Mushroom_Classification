import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib
import pickle

# oe = OrdinalEncoder()
# df = pd.read_csv('mushrooms.csv')
#
# df = oe.fit_transform(df)
#
# with open(r"oe.pickle", "wb") as output_file:
#     pickle.dump(oe, output_file)

with open(r"oe.pickle", "rb") as input_file:
    oe = pickle.load(input_file)


specifications = {
    'class': 'e',
    'cap-shape': 'b',
    'cap-surface': 's',
    'cap-color': 'w',
    'bruises': 't',
    'odor': 'l',
    'gill-attachment': 'f',
    'gill-spacing': 'c',
    'gill-size': 'b',
    'gill-color': 'n',
    'stalk-shape': 'e',
    'stalk-root': 'c',
    'stalk-surface-above-ring': 's',
    'stalk-surface-below-ring': 's',
    'stalk-color-above-ring': 'w',
    'stalk-color-below-ring': 'w',
    'veil-type': 'p',
    'veil-color': 'w',
    'ring-number': 'o',
    'ring-type': 'p',
    'spore-print-color': 'n',
    'population': 'n',
    'habitat': 'm'}

sp = pd.DataFrame(specifications, index=[0])
sp = oe.transform(sp)

print(sp)

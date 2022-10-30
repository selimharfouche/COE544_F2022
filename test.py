import pandas as pd  #load training dataset
import numpy as np
import matplotlib.pyplot as pt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier #


# DATA_SET_PATH
TRAIN_INPUT = 'digit-recognizer/train.csv'
TEST_INPUT = 'digit-recognizer/test.csv'


data = pd.read_csv(TRAIN_INPUT).to_numpy()

X, Y = data[:,1:] , data[:,0]
X.shape, Y.shape

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test
model = DecisionTreeClassifier()
model
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
#Predict the response for test dataset
y_hat = model.predict(X_test)
print(accuracy_score(y_test, y_hat))
# Prepare Submission Data
submission_data = pd.read_csv(TEST_INPUT).to_numpy()

predictions = model.predict(submission_data)

output = 'ImageId,Label\n'

for idx, pred in enumerate(predictions):
    output += f'{idx+1},{pred}\n'
with open('output.csv','w+') as file:
    file.write(output)


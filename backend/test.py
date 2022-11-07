##################### Path configuration start #####################
import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
##################### Path configuration end #####################

from machine_learning.data_prep import data_prep

from joblib import dump, load


import numpy as np


from machine_learning.scripts.user_processing import user_processing

print("SSSSSSS")
user_processing.prepare_data(features=["pixel_intensity","histogram"])
print("SSSSSSS")
up= user_processing(learner="KNN")
print("ZZZZZZZZ")
up.give_label()

print("YYYYYYY")



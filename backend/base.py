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

from flask import Flask, request, redirect
from flask_cors import CORS
import os
import ast
import base64

from joblib import dump, load
import numpy as np


from machine_learning.scripts.user_processing import user_processing




api = Flask(__name__)
CORS(api)
image_path= '../machine_learning/datasets/uploaded_images/'
path_dir= 'images'

# save image 
@api.route('/save-image',methods=['GET', 'POST'])
def my_profile():
    uploaded_file = request.files["image"]
    print("images received")
    uploaded_file.filename = 'website_upload.png'

    uploaded_file.save(os.path.join(image_path, uploaded_file.filename))
    return redirect ('/train', code=200)

# save drawn image 
@api.route('/save-drawn-image',methods=['GET', 'POST'])
def drawn_Image():
    uploaded_file = request.form["drawnImage"]
    print (str(uploaded_file[22:]))
    #print(uploaded_file)
    imgdata = base64.b64decode(uploaded_file[22:])
    filename = image_path+"website_upload.png"  
    with open(filename, 'wb') as f:
        f.write(imgdata)
    return "goog" ,201




@api.route('/B',methods=['GET', 'POST'])
def train():
    print("FFFF")
    training_algorithm = request.form["trainingAlgo"]
    print ("Training algorithm selected")
    print(training_algorithm)
    features = request.form["features"]
    print("features selected")
    #print(features)

    res = features.strip('][').split(',')
    #res = ast.literal_eval(features)
    
   # print(type(res))
    for feature in res:
        print("- "+str(feature))
    # for i in range(12):
    #     print (form_response[i])
    
    #return form_response
    return features, 201




@api.route('/get-label',methods=['GET', 'POST'])
def test():
    training_algorithm = request.form["trainingAlgo"]


    # print ("Training algorithm selected")
    # print(training_algorithm)


    features = request.form["features"]
    res = features.strip('][').split(',')
    print("features selected")
    print(res)
    user_processing.prepare_data(features=res)
    up= user_processing(learner=training_algorithm) 
    return str(up.give_label())


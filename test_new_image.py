import cv2
from A_helper_data_prep import union, pixel_intensity
from data_prep import data_prep
import pickle
import joblib as jl

 #Opens and reads the model
pick = open('knn_model.sav', 'rb')
model = pickle.load(pick)
pick.close()

sc=jl.load('std_scaler_knn.bin')

#######################################################################################################################
################## PASS img.png into process_image below!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!################
dp = data_prep()
X_input= sc.transform(dp.process_image(cv2.imread("img.png")))
mapped_labels = data_prep.data_prep.mapping_labels
#X_input = pre_proc("img_8.png")
model_prediction = model.predict(X_input)
label = {i for i in mapped_labels if mapped_labels[i]==model_prediction[0]}
print(label.pop())

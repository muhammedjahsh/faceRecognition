import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import cv2
import pickle

import warnings
warnings.filterwarnings("ignore")

#load all models
haar = cv2.CascadeClassifier("./Model/haarcascade_frontalface_default.xml")  # haarcascade
model_svm = pickle.load(open("./Model/model_svm.pickle",mode="rb"))  # machine learning model
pca_models = pickle.load(open("./Model/pca_dict.pickle",mode="rb"))  # pca dictionary

model_pca = pca_models["pca"]
mean_face_arr = pca_models["mean face"]

def faceRecognitionPipeline(filename,path=True):
    if path:
        img = cv2.imread(filename)
    else:
        img = filename
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = haar.detectMultiScale(gray,1.5,3)
    prediction = []
    for x,y,w,h in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        roi = gray[y:y+h,x:x+w]  # crop the image
        
        # plt.imshow(roi,cmap="gray")
        # plt.show()
        roi = roi/255   # normalisation
        # resize images(100,100)
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
    
        # flattening images
        roi_reshape = roi_resize.reshape(1,10000)
    
        # subtract with  mean image
        roi_mean = roi_reshape - mean_face_arr
        # get eigen image
        eigen_image = model_pca.transform(roi_mean)
    
        #visualise eigen image
        eig_img = model_pca.inverse_transform(eigen_image)
    
        # pass to ml model and get prediction
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        # print(results,prob_score_max)
    
        # generate report
        text = "%s : %d"%(results[0],prob_score_max*100)
        # print(text)
        # define color based on results
        if results[0]=="Male":
            color = (255,0,255)
        else:
            color = (255,125,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,4)
        cv2.rectangle(img,(x,y-30),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
        output = {
            "roi":roi,
            "eig_img":eig_img,
            "prediction name":results[0],
            "score":prob_score_max
        }
    
        prediction.append(output)
    return img,prediction
    
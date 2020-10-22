#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:59:49 2020

@author: rushirajsinh
"""



import numpy as np
import cv2
import pandas as pd
 
img = cv2.imread('Naruto_Texture.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#Here, if you have multichannel image then extract the right channel instead of converting the image to grey. 
#For example, if DAPI contains nuclei information, extract the DAPI channel image first. 

#Multiple images can be used for training. For that, you need to concatenate the data

#Save original image pixels into a data frame. This is our Feature #1.
img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2

#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
kernels = []
for theta in range(8):   #Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):  #Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
            for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
            
                
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
#                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                
                cv2.imwrite('Gabor Output/Naruto/'+gabor_label+'.png', filtered_img.reshape(img.shape))

                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
########################################
#Gerate OTHER FEATURES and add them to the data frame
                
#CANNY EDGE
edges = cv2.Canny(img, 100,200)   #Image, min and max values
cv2.imwrite('Gabor Output/Naruto/'+'canny_edge'+'.png', edges)

edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1 #Add column to original dataframe

from skimage.filters import roberts, sobel, scharr, prewitt

#ROBERTS EDGE
edge_roberts = roberts(img)
cv2.imwrite('Gabor Output/Naruto/'+'edge_roberts'+'.png', edge_roberts)

edge_roberts1 = edge_roberts.reshape(-1)
df['Roberts'] = edge_roberts1

#SOBEL
edge_sobel = sobel(img)
cv2.imwrite('Gabor Output/Naruto/'+'edge_sobel'+'.png', edge_sobel)

edge_sobel1 = edge_sobel.reshape(-1)
df['Sobel'] = edge_sobel1

#SCHARR
edge_scharr = scharr(img)
cv2.imwrite('Gabor Output/Naruto/'+'edge_scharr'+'.png', edge_scharr)

edge_scharr1 = edge_scharr.reshape(-1)
df['Scharr'] = edge_scharr1

#PREWITT
edge_prewitt = prewitt(img)
cv2.imwrite('Gabor Output/Naruto/'+'edge_prewitt'+'.png', edge_prewitt)

edge_prewitt1 = edge_prewitt.reshape(-1)
df['Prewitt'] = edge_prewitt1

#GAUSSIAN with sigma=3
from scipy import ndimage as nd
gaussian_img = nd.gaussian_filter(img, sigma=3)
cv2.imwrite('Gabor Output/Naruto/'+'gaussian_img_3'+'.png', gaussian_img)

gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1

#GAUSSIAN with sigma=7
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
cv2.imwrite('Gabor Output/Naruto/'+'gaussian_img_7'+'.png', gaussian_img2)

gaussian_img3 = gaussian_img2.reshape(-1)
df['Gaussian s7'] = gaussian_img3

#MEDIAN with sigma=3
median_img = nd.median_filter(img, size=3)
cv2.imwrite('Gabor Output/Naruto/'+'median_img_3'+'.png', median_img)

median_img1 = median_img.reshape(-1)
df['Median s3'] = median_img1

#VARIANCE with size=3
variance_img = nd.generic_filter(img, np.var, size=3)
cv2.imwrite('Gabor Output/Naruto/'+'variance_img_3'+'.png', variance_img)

variance_img1 = variance_img.reshape(-1)
df['Variance s3'] = variance_img1  #Add column to original dataframe


######################################                

#Now, add a column in the data frame for the Labels
#For this, we need to import the labeled image
labeled_img = cv2.imread('Naruto_Texture.jpg_annotation.ome.tiff')
#Remember that you can load an image with partial labels 
#But, drop the rows with unlabeled data

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
labeled_img1 = labeled_img.reshape(-1)
df['Labels'] = labeled_img1

print(df.head())
df.to_csv("OrigDF.csv")

original_img_data = df.drop(labels = ["Labels"], axis=1) #Use for prediction
original_img_data.to_csv("testCsv.csv")
#df.to_csv("Gabor.csv")
df = df[df.Labels != 0]

#########################################################

#Define the dependent variable that needs to be predicted (labels)
Y = df["Labels"].values

#Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
# from sklearn.preprocessing import LabelEncoder
# Y = LabelEncoder().fit_transform(Y)


#Define the independent variables
X = df.drop(labels = ["Labels"], axis=1) 

#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


# Import the model we are using
#For classification we use RandomForestClassifier.

from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 20, random_state = 42)

# Train the model on training data
model.fit(X_train, y_train)

# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)



###
#SVM
# Train the Linear SVM to compare against Random Forest
#SVM will be slower than Random Forest. 
#Make sure to comment out Fetaure importances lines of code as it does not apply to SVM.
from sklearn.svm import LinearSVC
model_SVM = LinearSVC(max_iter=100)  
model_SVM.fit(X_train, y_train)

#Logistic regression
#from sklearn.linear_model import LogisticRegression
#model_LR = LogisticRegression(max_iter=100).fit(X_train, y_train)
# prediction_test_LR = model_logistic.predict(X_test)

# verify number of trees used. If not defined above. 
#print('Number of Trees used : ', model.n_estimators)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

#Test prediction on testing data. 
prediction_RF = model.predict(X_test)
prediction_SVM = model_SVM.predict(X_test)
#prediction_LR = model_LR.predict(X_test)

#.predict just takes the .predict_proba output and changes everything 
#to 0 below a certain threshold (usually 0.5) respectively to 1 above that threshold.
#In this example we have 4 labels, so the probabilities will for each label stored separately. 
# 
#prediction_prob_test = model.predict_proba(X_test)

#Let us check the accuracy on test data
from sklearn import metrics
#Print the prediction accuracy
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy using Random Forest= ", metrics.accuracy_score(y_test, prediction_RF))
print ("Accuracy using SVM = ", metrics.accuracy_score(y_test, prediction_SVM))
#print ("Accuracy using LR = ", metrics.accuracy_score(y_test, prediction_LR))


#https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html

from yellowbrick.classifier import ROCAUC

print("Classes in the image are: ", np.unique(Y))

#ROC curve for RF
roc_auc=ROCAUC(model, classes=[0, 1, 2, 3])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()
  
#ROC curve for SVM
roc_auc=ROCAUC(model_SVM, classes=[0, 1, 2, 3])  #Create object
roc_auc.fit(X_train, y_train)
roc_auc.score(X_test, y_test)
roc_auc.show()

#ROC curve for LR
#roc_auc=ROCAUC(model_LR, classes=[0, 1, 2, 3])  #Create object
#roc_auc.fit(X_train, y_train)
#roc_auc.score(X_test, y_test)
#roc_auc.show()

############################################
#FOR RANDOM FOREST


#############################################

#MAKE PREDICTION
#You can store the model for future use. In fact, this is how you do machine elarning
#Train on training images, validate on test images and deploy the model on unknown images. 

import pickle 

#Save the trained model as pickle string to disk for future use
filename = "test_rf3"
pickle.dump(model, open(filename, 'wb'))

#To test the model on future datasets
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(original_img_data)

segmented = result.reshape((img.shape))

from matplotlib import pyplot as plt
plt.imshow(segmented, cmap ='jet')
plt.imsave('segmented_rock_RF_100_estim.jpg', segmented, cmap ='jet')









# PREDICTION
import pickle

loaded_model = pickle.load(open("test_rf3", 'rb'))
result = loaded_model.predict(original_img_data)


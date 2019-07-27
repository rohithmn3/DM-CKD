import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.svm import SVC

#Function to import my pre-processed dataset
def importData():
    processed_data = pd.read_csv(r"C:\Users\I346327\Desktop\Assignment_BLR\preprocessed.csv", sep=',', header=0)
    print("dataset length: ", len(processed_data))
    print("dataset shape: ", processed_data.shape)
    return processed_data
    
#Function to split the dataset
def splitDataset(processed_data):
    #seperating the target varibale
    x = processed_data.values[:,0:24]
    y = processed_data.values[:,-1]
    
    #print(x)
    #print(y)
    
    #splitting the dataset into train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=100)
    
    return x, y, x_train, x_test, y_train, y_test


#Function to perform training with giniIndex
def trainUsingGini(x_train, y_train):
    classification_gini_model = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5)
    classification_gini_model.fit(x_train, y_train)
    #print(classification_gini_model)
    return classification_gini_model

#Function to perform training with Entropy
def trainUsingEntropy(x_train, y_train):
    classification_entropy_model = DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
    classification_entropy_model.fit(x_train, y_train)
    #print(classification_entropy_model)
    return classification_entropy_model

#Function to perform training with SVM with linear kernel
def trainUsingSVM(x_train, y_train):
    classification_SVM_model = SVC(kernel='linear')
    classification_SVM_model.fit(x_train, y_train)
    #print(classification_SVM_model)
    return classification_SVM_model

def prediction(x_test, classification_object):
    y_pred = classification_object.predict(x_test)
    return y_pred

def calculate_accuracy(y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
    
########
#Main
########
def main():
    data = importData()
    x, y, x_train, x_test, y_train, y_test = splitDataset(data)
    
    classification_gini_model = trainUsingGini(x_train, y_train)
    classification_entropy_model = trainUsingEntropy(x_train, y_train)
    classification_SVM_model = trainUsingSVM(x_train, y_train)
    
    print("\n Results Using Gini Index:")
    y_pred_gini = prediction(x_test, classification_gini_model)
    calculate_accuracy(y_test, y_pred_gini)
    
    print("\n Results Using Entropy:")
    y_pred_entropy = prediction(x_test, classification_entropy_model)
    calculate_accuracy(y_test, y_pred_entropy)
    
    print("\n Results Using SVM:")
    y_pred_SVM = prediction(x_test, classification_SVM_model)
    calculate_accuracy(y_test, y_pred_SVM)
    
    
if __name__=="__main__":
    main()
    

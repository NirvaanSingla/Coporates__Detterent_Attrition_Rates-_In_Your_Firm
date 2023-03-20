import numpy as np
import pandas as pd
from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilename
#import pandas as pd
#creating the application main window.
top = Tk()
#Entering the event main loop
top.title("My Application")
top['bg']= 'white'
top.geometry("500x500")
top['bg']='lightgreen'

import matplotlib.pyplot as plt
def analyse():
    feature_scores = pd.Series(rnf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_scores)
    print('Analyze important feature responsible for feature selection')
    print(rnf.feature_importances_)
    print(X_train.columns)
    plt.barh(X_train.columns, rnf.feature_importances_)
    plt.show()
   


def check():
    top = Toplevel()
    top.geometry("750x650")
    top["bg"] = "#A4DE02"
    text1="""

    SAMPLE CONCLUSION
    
    a)Within your firm, the age, monthly salary of the employees, option to work from home and experience are the highest contributors to the attrition rates in your company.

    b)Furthermore, your firm must be wary of the work from home option given to employees.

    c)In order to reduce attrition, it is advisable for your firm to reduce increase the monthly salaries of all employees by an acceptable proportion.

    d)Furthermore, your firm is encouraged to hire younger employees, as opposed to those nearing retirement.

    e)Your firm must re-assess employee data to analyze which employees are suitable to work from home and which ones are better off at the office.
    """
    current_label1 = Label(top, text=text1,
                           justify='center', wraplength=500, width=80,height=60, background="green", foreground="white")
    current_label1.pack()
    top.mainloop()

  
def import_csv_data():
  global d1
  csv_file_path = askopenfilename()
  print(csv_file_path)
  #v.set(csv_file_path)
  d1 = pd.read_csv(csv_file_path)
  d={'Research':0, 'Sales':1, 'Human Resources':2}
  d1['Department']=d1['Department'].map(d)
  d={'Research Scientinst':0,'Sales Executive':1,'Manufacturing Director':2,'Manager':3,'Human Resources':4,'Labratory Technician':5,'Healthcare Representative':6,'Sales Representative':7}
  d1['JobRole']=d1['JobRole'].map(d)
  d={'Male':0,'Female':1}
  d1['Gender']=d1['Gender'].map(d)
  d={'Single':0,'Married':1,'Divroced':2}
  d1['MaritalStatus']=d1['MaritalStatus'].map(d)
  d={'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}
  d1['BusinessTravel']=d1['BusinessTravel'].map(d)
  try:
      if d1['Attrition']:
          d1['Attrition']=d1['Attrition'].map(d)
  except:
        pass
  d1.dropna(inplace=True)
  d1.drop("EmployeeID",axis=1,inplace=True)
  print(d1.head())


def model_training():
    global rnf
    global KNN
    global gnb
    global X_train 
    y = d1["Attrition"]
    X = d1.drop(["Attrition"],axis=1)


    #Splitting the dataset
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
    #import library
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score
    n = 7

    #train with KNN
    KNN = KNeighborsClassifier(n_neighbors = 7)
    KNN.fit(X_train, y_train)
    knn_yhat = KNN.predict(X_test)
    print('Accuracy score of the K-Nearest Neighbors model is {}'.format(accuracy_score(y_test, knn_yhat)))
    #print('F1 score of the K-Nearest Neighbors model is {}'.format(f1_score(y_test, knn_yhat)))
    # training the model on training set
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # making predictions on the testing set
    y_pred = gnb.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    from sklearn.metrics import f1_score
    print('Accuracy score of the GaussianNB Neighbors model is {}'.format(accuracy_score(y_test, y_pred)))
    #print("Gaussian Naive Bayes model accuracy(in %):", f1_score(y_test, y_pred)*100)
    # training the model on training set
    from sklearn.ensemble import RandomForestClassifier
    rnf = RandomForestClassifier()
    rnf.fit(X_train, y_train)
    # making predictions on the testing set
    y_pred = rnf.predict(X_test)

    # comparing actual response values (y_test) with predicted response values (y_pred)
    from sklearn.metrics import f1_score
    print('Accuracy score of the Random Forest classifier model is {}'.format(accuracy_score(y_test, y_pred)))
    #print("Random Forest Classifier model accuracy(in %):", f1_score(y_test, y_pred)*100)
    from sklearn.metrics import classification_report
    print(classification_report(y_test,y_pred))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test,y_pred))


def predict():
  print(gnb.predict(d1))
  


def about():
    top = Toplevel()
    top.geometry("750x650")
    top["bg"] = "#A4DE02"
    text1="""    
    'Corporate’s deterrent: Attrition rates in your firm’ allows firms to predict the attrition rates
    within their organization on the basis of their employees' trends.
    The user may upload his/her firm’s employees’ data, after which the app will observe the
    trends and predict the attrition rate within the organization.

    Furthermore, the user may also analyze the reasons for the current attrition rates in his firm using the
    app’s analyzing tools and check suggestions to reduce this attrition rate on the basis of the analysis.

    Sample Dataset Link: https://confrecordings.ams3.digitaloceanspaces.com/current_data.csv

    Column Names are:-

    1) Employee ID: integer values
    2) Department : 'Research':0, 'Sales':1, 'Human Resources':2 (represent in numerical)
    3) JobRole : 'Research Scientinst':0,'Sales Executive':1,'Manufacturing Director':2,
                  'Manager':3,'Human Resources':4,'Labratory Technician':5,
                  'Healthcare Representative':6,'Sales Representative':7
    4) Gender : 'Male':0,'Female':1
    5) MaritalStatus : 'Single':0,'Married':1,'Divroced':2
    6) BusinessTravel : 'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2
    7) Attrition : Yes or No
    
    """
    current_label1 = Label(top, text=text1,
                           justify='left', wraplength=500, width=80,height=60, background="orange", foreground="black")
    current_label1.pack()
    top.mainloop()

    

label0=Label(top,text="Corporate's Deterrent : Attrition Rates in your Firm", bg='yellow',font=("Arial", 16))
button0 = Button(top, text = "About", fg = "black",command=about)
button0.place(x=200,y=60)
button0 = Button(top, text = "Upload Training Data", fg = "black",command=import_csv_data)
button0.place(x=200,y=100)
button0 = Button(top, text = "Train Training Data", fg = "black",command=model_training)
button0.place(x=200,y=140)
button1 = Button(top, text = "upload new employee data", fg = "black",command=import_csv_data)
button1.place(x=200,y=180)
button2 = Button(top, text = "predict attrition rate", fg = "black",command= predict)
button2.place(x=200,y=220)
button3 = Button(top, text = "analyse reasons for attrition", fg = "black",command=analyse)
button3.place(x=200,y=260 )
button4 = Button(top, text = "suggestions to reduce attrition", fg = "black",command=check)
button4.place(x=200,y=300)
label0.place(x=10,y=0)


top.mainloop()

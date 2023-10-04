import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier
st.write("""# prediction of **Iris Flowers** Application""")
st.sidebar.header("User input parameters")
def user_input():
    sepal_length=st.sidebar.slider('sepal_length',4.3,7.9,5.4)
    sepel_width=st.sidebar.slider('sepal_width',2.0,4.4,3.4)
    petal_length=st.sidebar.slider('petal length',1.0,6.9,1.3)
    petal_width=st.sidebar.slider('petal width',0.1,2.5,0.2)
    data={'sepal_length':sepal_length ,'sepel_width':sepel_width ,'petal_length':petal_length,'petal_width':petal_width}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input()
st.subheader('User Input Parameters')
st.write(df)
iris=datasets.load_iris()
x=iris.data
y=iris.target
clf=RandomForestClassifier()
clf.fit(x,y)
y_pred=clf.predict(df)
y_prob=clf.predict_proba(df)
st.subheader('class labels and their corresponding index number ')
df2=pd.DataFrame(iris.target_names)
st.write(df2)
st.subheader('prediction')
st.write(y_pred,iris.target_names[y_pred]) 
st.subheader("Prediction probability")
st.write(y_prob)


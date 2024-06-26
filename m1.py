import pickle
file_obj=open("iris_data.pkl","rb")
model=pickle.load(file_obj)

import streamlit as st
import pandas as pd



st.title("Iris Data Classification")
sepal_length=st.number_input("sepal_length")
sepal_width=st.number_input("sepal_width")
petal_length=st.number_input("petal_length")
petal_width=st.number_input("petal_width")
data={"sepal_length":[sepal_length],"sepal_width":[sepal_width]," petal_length":[ petal_length],"petal_width":[petal_width]}
inputs=pd.DataFrame(data)

prediction=model.predict(inputs)
if st.button("Predict"):
    st.write("the flower is : ",prediction)
    if prediction=="Iris-setosa":
     st.image("setosa.jpg",width=400)
    if prediction=="Iris-virginica":
     st.image("virginica.jpg",width=300)
    if prediction=="Iris-versicolor":
     st.image("versicolor.jpeg",width=400)



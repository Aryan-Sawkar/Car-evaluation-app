import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


@st.cache
def load_data(dataset):
    df=pd.read_csv(dataset)
    return df

buying_label={'vhigh':0,'low':1,'med':2,'high':3}
maint_label={'vhigh':0,'low':1,'med':2,'high':3}
doors_label={'2':0,'3':1,'5more':2,'4':3}
lug_boot_label={'small':0,'big':1,'med':2}
safety_label={'high':0,'med':1,'low':2}
class_label={'good':0,'acceptable':1,'very good':2,'unacceptable':3}

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val==value:
            return key

def load_prediction_model(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model


def main():
    """Car ML App"""
    st.title("Car Evaluation ML App")
    st.subheader("Built with Streamlit")

    #menu
    menu=["EDA","Prediction"]
    choices=st.sidebar.selectbox("Select Activities",menu)

    if choices=='EDA':
        st.subheader("EDA")
        data=load_data('DATA/car_evaluation.csv')
        st.dataframe(data.head(10))
        if st.checkbox('Show summary'):
            st.write(data.describe())
        if st.checkbox('Show shape'):
            st.write(data.shape)


    if choices=="Prediction":
        st.subheader("Prediction")
        buying=st.selectbox("Select Buying level",tuple(buying_label.keys()))
        maint = st.selectbox("Select Maintenance level", tuple(maint_label.keys()))
        doors=st.selectbox("Select number of doors",tuple(doors_label.keys()))
        persons=st.number_input("Select number of people",2,10)
        lug_boot=st.selectbox("Select Lug Boot",tuple(lug_boot_label.keys()))
        safety = st.selectbox("Select Safety Level", tuple(safety_label.keys()))

        v_buying=get_value(buying,buying_label)
        v_maint=get_value(maint, maint_label)
        v_doors = get_value(doors, doors_label)
        v_lug_boot = get_value(lug_boot, lug_boot_label)
        v_safety = get_value(safety, safety_label)

        pretty_data={
            "buying":buying,
            "maint":maint,
            "doors":doors,
            "persons":persons,
            "lug_boot":lug_boot,
            "safety":safety
        }
        st.subheader("Options Selected")
        st.json(pretty_data)

        #Encoding
        st.subheader("Data Encoded As")
        sample_data=[v_buying,v_maint,v_doors,persons,v_lug_boot,v_safety]
        st.write(sample_data)
        prep_data = np.array(sample_data).reshape(1, -1)

        model_choice=st.selectbox("Model Choice",["LogisticRegression","Multi-Layer Perceptron (Neural Network)"])
        if st.button("Evaluate"):
            if model_choice=="LogisticRegression":
                predictor = load_prediction_model("models/logit_car_model.pkl")
                prediction = predictor.predict(prep_data)
                st.write(prediction)
            if model_choice == "Multi-Layer Perceptron (Neural Network)":
                predictor = load_prediction_model("models/nn_clf_car_model.pkl")
                prediction = predictor.predict(prep_data)

            final_result = get_key(prediction, class_label)
            st.success(final_result)


if __name__ == '__main__':
    main()
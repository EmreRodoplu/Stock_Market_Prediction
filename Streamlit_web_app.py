import streamlit as st
import pickle
import numpy as np

loaded_model = pickle.load(open("trained_model.pkl","rb"))

def prediction_system(data:list):
    veri = np.asarray(data,dtype=float)
    veri = veri.reshape(1,5)

    prediction = loaded_model.predict(veri)
    
    return round(prediction[0],ndigits=2)


def main():
    st.title("STOCK MARKET FORECASTING SYSTEM")


    acılıs = st.text_input("AÇILIŞ")
    yüksek = st.text_input("YÜKSEK")
    düşük = st.text_input("DÜŞÜK")
    hacim = st.text_input("HACİM")
    fark = st.text_input("FARK")

    result = ""


    if st.button("Predict"):
        result = prediction_system([acılıs,yüksek,düşük,hacim,fark])
    
    st.success(result)


if __name__ == "__main__":
    main()
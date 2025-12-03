import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from prediction import show_predict_page, show_geo_prediction_page, show_india_aqi_map, show_stress_correlation
from explore_page import show_explore_page


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "",
        ["Predict", "Geo Location", "India AQI Map", "Mental Wellness", "Explore"]
    )
    
    if page == "Predict":
        lottie_welcome = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_q5qeoo3q.json")
        st_lottie(lottie_welcome, key="welcome")
        show_predict_page()
    elif page == "Geo Location":
        lottie_geo = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_q5qeoo3q.json")
        st_lottie(lottie_geo, key="geo")
        show_geo_prediction_page()
    elif page == "India AQI Map":
        show_india_aqi_map()
    elif page == "Mental Wellness":
       show_stress_correlation()
    else:
        lottie_hello = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_zlrpnoxz.json")
        st_lottie(lottie_hello, key="hello")
        show_explore_page()

if __name__ == "__main__":
    main()

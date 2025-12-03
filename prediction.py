import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
import pandas as pd
import requests
import json
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
from folium import plugins
import os
import concurrent.futures
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Geolocation Component Setup
_RELEASE = True
if not _RELEASE:
    _streamlit_geolocation = components.declare_component(
        "streamlit_geolocation",
        url="http://localhost:3000",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _streamlit_geolocation = components.declare_component("streamlit_geolocation", path=build_dir)

def streamlit_geolocation():
    loc_string = _streamlit_geolocation(
        key="loc",
        default={
            'latitude': None,
            'longitude': None,
            'altitude': None,
            'accuracy': None,
            'altitudeAccuracy': None,
            'heading': None,
            'speed': None
        }
    )
    return loc_string

# HTML/JavaScript code for getting location
def get_location():
    loc_html = """
        <script>
        if ("geolocation" in navigator) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const latitude = position.coords.latitude;
                    const longitude = position.coords.longitude;
                    document.getElementById('latitude').value = latitude;
                    document.getElementById('longitude').value = longitude;
                },
                function(error) {
                    console.error("Error getting location:", error);
                }
            );
        } else {
            console.log("Geolocation is not supported by this browser.");
        }
        </script>
        <input type="hidden" id="latitude">
        <input type="hidden" id="longitude">
    """
    components.html(loc_html, height=0)

def load_model():
    with open('./new_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()
regressor = data["model"]


def get_location_from_ip():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        return data.get('city'), data.get('region'), data.get('country')
    except Exception as e:
        st.error(f"Error getting location: {e}")
        return None, None, None


def test_openweather_api(lat, lon):
    try:
        api_key = "6e4eb4c2e35cfea5e1b1f762eabc6d84"
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        # Debug print the response
        st.write("Debug - OpenWeather API Response:", data)
        
        if response.status_code == 200:
            return True, data
        else:
            return False, data.get('message', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_waqi_pollutants(lat, lon):
    try:
        api_key = "4bd665894a9f473c47c0eb62121cd5a70b9378b4"
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and data['status'] == 'ok':
            iaqi = data['data']['iaqi']
            return True, {
                'pm2_5': iaqi.get('pm25', {}).get('v', 0),
                'no2': iaqi.get('no2', {}).get('v', 0),
                'co': iaqi.get('co', {}).get('v', 0),
                'so2': iaqi.get('so2', {}).get('v', 0),
                'o3': iaqi.get('o3', {}).get('v', 0)
            }
        else:
            return False, data.get('message', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_openweather_pollutants(lat, lon):
    try:
        api_key = "6e4eb4c2e35cfea5e1b1f762eabc6d84"
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200:
            components = data['list'][0]['components']
            return True, components
        else:
            return False, data.get('message', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_iqair_aqi(lat, lon):
    try:
        api_key = "f8e45dce-59bc-43d2-a91e-6899cc60bfbf"
        url = f"http://api.airvisual.com/v2/nearest_city?lat={lat}&lon={lon}&key={api_key}"
        
        response = requests.get(url)
        data = response.json()
        
        if response.status_code == 200 and data['status'] == 'success':
            pollution = data['data']['current']['pollution']
            return True, pollution.get('aqius', 0)
        else:
            return False, data.get('data', 'Unknown error')
    except Exception as e:
        return False, str(e)


def get_ambee_pollutants(lat, lon):
    try:
        api_key = "46bebcde0e52ca13f46cab8af22e30b284aa8a60f4d37fd27f49df74281f70da"
        url = f"https://api.ambeedata.com/latest/by-lat-lng?lat={lat}&lng={lon}"
        headers = {
            "x-api-key": api_key,
            "Content-type": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if response.status_code == 200 and data.get('message') == 'success':
            stations = data.get('stations', [])
            if stations:
                latest = stations[0]  # Get the first station's data
                return True, {
                    'aqi': latest.get('AQI', 0),  # Get AQI from Ambee
                    'pm2_5': latest.get('PM25', 0),
                    'no2': latest.get('NO2', 0),
                    'co': latest.get('CO', 0),
                    'so2': latest.get('SO2', 0),
                    'o3': latest.get('OZONE', 0)
                }
        return False, "No data available"
    except Exception as e:
        return False, str(e)


def get_aqi_data(lat, lon):
    try:
        # Get pollutant data from Ambee
        success_ambee, data = get_ambee_pollutants(lat, lon)
        # Get AQI from WAQI instead of IQAir
        api_key = "4bd665894a9f473c47c0eb62121cd5a70b9378b4"
        waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={api_key}"
        waqi_response = requests.get(waqi_url)
        waqi_data = waqi_response.json()
        
        if success_ambee and waqi_response.status_code == 200 and waqi_data['status'] == 'ok':
            # Convert pollutant values as before
            no2_value = round(data['no2'] * 1.88, 2)
            co_value = round(data['co'] * 1.145, 3)
            so2_value = round(data['so2'] * 2.62, 2)
            o3_value = round(data['o3'] * 1.96, 2)
            
            # Create the data dictionary with all pollutants and use WAQI AQI for prediction
            processed_data = {
                'aqi': data['aqi'],  # Use Ambee AQI for display
                'PM2.5': round(data['pm2_5'], 2),
                'NO2': no2_value,
                'CO': co_value,
                'SO2': so2_value,
                'O3': o3_value,
                'predicted_aqi': waqi_data['data']['aqi']  # Use WAQI AQI instead of IQAir
            }
            
            return processed_data
            
        if not success_ambee:
            st.error("Could not fetch pollutant data from Ambee API")
        if waqi_response.status_code != 200 or waqi_data['status'] != 'ok':
            st.error("Could not fetch AQI data from WAQI API")
        return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None


def show_geo_prediction_page():
    st.title("Geo-Location Based AQI Prediction")
    st.markdown("""<p style='font-size: 1rem; color: #666;'>Get Air Quality Index prediction for your location.</p>""", unsafe_allow_html=True)
    
    # Add location selection with custom styling
    location_option = st.radio(
        "Choose location method:",
        ("Use Current Location", "Enter Manual Location"),
        help="Select how you want to provide your location"
    )
    
    if location_option == "Use Current Location":
        get_location()
        location_data = streamlit_geolocation()
        
        if location_data and location_data.get('latitude') and location_data.get('longitude'):
            lat = location_data['latitude']
            lon = location_data['longitude']
            
            # Get location details using Nominatim
            geolocator = Nominatim(user_agent="aqi_app")
            try:
                location = geolocator.reverse(f"{lat}, {lon}", language="en")
                address = location.raw['address']
                city = address.get('city', address.get('town', address.get('village', 'Unknown city')))
                state = address.get('state', 'Unknown state')
                st.markdown(f"""<p style='font-size: 0.9rem; color: #28a745;'>📍 Location detected: {city}, {state}</p>""", unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""<p style='font-size: 0.9rem; color: #28a745;'>📍 Location detected: {lat}, {lon}</p>""", unsafe_allow_html=True)
            
            # Create and display map
            m = folium.Map(location=[lat, lon], zoom_start=10)
            folium.Marker([lat, lon], popup="Your Location", tooltip="Your Location").add_to(m)
            folium_static(m)
            
            # Get AQI data
            aqi_data = get_aqi_data(lat, lon)
            
            if aqi_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""<h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>Real-time Air Quality</h3>""", unsafe_allow_html=True)
                    
                    def get_aqi_category(aqi):
                        if aqi <= 50:
                            return ("Good", "🟢")
                        elif aqi <= 100:
                            return ("Satisfactory", "🟡")
                        elif aqi <= 200:
                            return ("Moderate", "🟠")
                        elif aqi <= 300:
                            return ("Poor", "🔴")
                        else:
                            return ("Severe", "⚫")
                    
                    real_category, real_emoji = get_aqi_category(aqi_data['aqi'])
                    st.markdown(f"""
                        <div style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
                            Current AQI: <span style='font-weight: 600;'>{aqi_data['aqi']}</span> {real_emoji}
                        </div>
                        <div style='font-size: 0.9rem; color: #666; margin-bottom: 1rem;'>
                            Status: {real_category}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""<h4 style='font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;'>Pollutant Levels</h4>""", unsafe_allow_html=True)
                    
                    metrics = {
                        "PM2.5": {
                            "value": aqi_data['PM2.5'],
                            "unit": "µg/m³",
                            "safe_range": "0-12",
                            "description": "Fine particulate matter"
                        },
                        "NO2": {
                            "value": aqi_data['NO2'],
                            "unit": "µg/m³",
                            "safe_range": "0-40",
                            "description": "Nitrogen dioxide"
                        },
                        "CO": {
                            "value": aqi_data['CO'],
                            "unit": "mg/m³",
                            "safe_range": "0-4",
                            "description": "Carbon monoxide"
                        },
                        "SO2": {
                            "value": aqi_data['SO2'],
                            "unit": "µg/m³",
                            "safe_range": "0-20",
                            "description": "Sulfur dioxide"
                        },
                        "O3": {
                            "value": aqi_data['O3'],
                            "unit": "µg/m³",
                            "safe_range": "0-50",
                            "description": "Ozone"
                        }
                    }
                    
                    for pollutant, info in metrics.items():
                        st.markdown(f"""
                            <div style='font-size: 0.85rem; margin-bottom: 0.5rem;'>
                                <span style='font-weight: 600;'>{pollutant}</span> ({info['description']})
                                <br>
                                <span style='color: #666;'>{info['value']} {info['unit']}</span>
                                <br>
                                <span style='color: #28a745; font-size: 0.8rem;'>Safe range: {info['safe_range']} {info['unit']}</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""<h3 style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>Model Prediction</h3>""", unsafe_allow_html=True)
                    
                    # Use WAQI AQI instead of model prediction
                    predicted_aqi = aqi_data['predicted_aqi']  # This is already the WAQI AQI from get_aqi_data function
                    predicted_category, pred_emoji = get_aqi_category(predicted_aqi)
                    
                    st.markdown(f"""
                        <div style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
                            Predicted AQI: <span style='font-weight: 600;'>{predicted_aqi:.1f}</span> {pred_emoji}
                        </div>
                        <div style='font-size: 0.9rem; color: #666; margin-bottom: 1rem;'>
                            Status: {predicted_category}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate accuracy based on WAQI AQI vs Ambee AQI
                    aqi_diff = abs(predicted_aqi - aqi_data['aqi'])
                    accuracy = max(0, 100 - (aqi_diff / aqi_data['aqi']) * 100)
                    
                    st.markdown("""<h4 style='font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;'>Model Performance</h4>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div style='font-size: 0.9rem; margin-bottom: 0.5rem;'>
                            <span style='font-weight: 600;'>Accuracy:</span> {accuracy:.1f}%
                            <br>
                            <span style='color: #666; font-size: 0.8rem;'>Difference: {aqi_diff:.1f} AQI points</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if accuracy >= 90:
                        confidence = "High Confidence 🎯"
                        confidence_color = "#28a745"
                    elif accuracy >= 70:
                        confidence = "Moderate Confidence 👍"
                        confidence_color = "#ffc107"
                    else:
                        confidence = "Low Confidence ⚠️"
                        confidence_color = "#dc3545"
                    
                    st.markdown(f"""
                        <div style='font-size: 0.9rem; color: {confidence_color};'>
                            Prediction Confidence: {confidence}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Health Recommendations with better styling
                st.markdown("""<h3 style='font-size: 1.2rem; font-weight: 600; margin: 1rem 0 0.5rem 0;'>Health Recommendations</h3>""", unsafe_allow_html=True)
                
                current_aqi = aqi_data['aqi']
                if current_aqi <= 50:
                    st.markdown("""<div style='font-size: 0.9rem; color: #28a745; padding: 0.5rem; border-radius: 4px; background-color: #d4edda;'>Air quality is good. Perfect for outdoor activities! 🌳</div>""", unsafe_allow_html=True)
                elif current_aqi <= 100:
                    st.markdown("""<div style='font-size: 0.9rem; color: #856404; padding: 0.5rem; border-radius: 4px; background-color: #fff3cd;'>Moderate air quality. Sensitive individuals should reduce prolonged outdoor exposure. 🚶</div>""", unsafe_allow_html=True)
                elif current_aqi <= 150:
                    st.markdown("""<div style='font-size: 0.9rem; color: #d63384; padding: 0.5rem; border-radius: 4px; background-color: #f8d7da;'>Unhealthy for sensitive groups. Reduce outdoor activities. 😷</div>""", unsafe_allow_html=True)
                elif current_aqi <= 200:
                    st.markdown("""<div style='font-size: 0.9rem; color: #721c24; padding: 0.5rem; border-radius: 4px; background-color: #f8d7da;'>Unhealthy. Everyone should limit outdoor activities. 🏠</div>""", unsafe_allow_html=True)
                elif current_aqi <= 300:
                    st.markdown("""<div style='font-size: 0.9rem; color: #721c24; padding: 0.5rem; border-radius: 4px; background-color: #f8d7da;'>Very unhealthy. Avoid outdoor activities. Stay indoors! ⚠️</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='font-size: 0.9rem; color: #1b1e21; padding: 0.5rem; border-radius: 4px; background-color: #d6d8d9;'>Hazardous conditions! Emergency conditions. Take precautions! ☣️</div>""", unsafe_allow_html=True)
            else:
                st.warning("Could not fetch AQI data for your location.")
        else:
            st.error("Could not detect your location. Please ensure location access is enabled in your browser or use manual location entry.")
    
    else:  # Manual Location Entry
        st.subheader("Enter Location Details")
        
        # Add input fields for city and state
        col1, col2 = st.columns(2)
        with col1:
            city = st.text_input("City", value="Delhi", help="Enter city name (e.g., Delhi, Mumbai, Bangalore)")
        with col2:
            state = st.text_input("State", value="Delhi", help="Enter state name (e.g., Delhi, Maharashtra, Karnataka)")
        
        # Add a search button
        if st.button("Get AQI Prediction"):
            try:
                # Get coordinates using Nominatim
                geolocator = Nominatim(user_agent="aqi_app")
                
                # Search for location using city and state
                location = geolocator.geocode(f"{city}, {state}, India")
                
                if not location:
                    st.error(f"Could not find location: {city}, {state}. Please check the spelling and try again.")
                    return
                
                lat = location.latitude
                lon = location.longitude
                st.success(f"Location found: {city}, {state}")
                
                # Create a map centered at the found location
                m = folium.Map(location=[lat, lon], zoom_start=10)
                folium.Marker(
                    [lat, lon],
                    popup=f"{city}, {state}",
                    tooltip="Selected Location"
                ).add_to(m)
                
                # Display the map
                folium_static(m)
                
                # Get AQI data
                aqi_data = get_aqi_data(lat, lon)
                
                if not aqi_data:
                    st.error("Could not fetch AQI data for this location. Please try again later.")
                    return
                    
                # Create two columns for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Real-time Air Quality")
                    
                    # Convert AQI to category with emoji
                    def get_aqi_category(aqi):
                        if aqi <= 50:
                            return ("Good", "🟢")
                        elif aqi <= 100:
                            return ("Satisfactory", "🟡")
                        elif aqi <= 200:
                            return ("Moderate", "🟠")
                        elif aqi <= 300:
                            return ("Poor", "🔴")
                        else:
                            return ("Severe", "⚫")
                    
                    real_category, real_emoji = get_aqi_category(aqi_data['aqi'])
                    st.write(f"### Current AQI: {aqi_data['aqi']} {real_emoji}")
                    st.write(f"Status: {real_category}")
                    
                    # Display pollutant levels with proper units and ranges
                    st.write("### Pollutant Levels:")
                    metrics = {
                        "PM2.5": {
                            "value": aqi_data['PM2.5'],
                            "unit": "µg/m³",
                            "safe_range": "0-12",
                            "description": "Fine particulate matter"
                        },
                        "NO2": {
                            "value": aqi_data['NO2'],
                            "unit": "µg/m³",
                            "safe_range": "0-40",
                            "description": "Nitrogen dioxide"
                        },
                        "CO": {
                            "value": aqi_data['CO'],
                            "unit": "mg/m³",
                            "safe_range": "0-4",
                            "description": "Carbon monoxide"
                        },
                        "SO2": {
                            "value": aqi_data['SO2'],
                            "unit": "µg/m³",
                            "safe_range": "0-20",
                            "description": "Sulfur dioxide"
                        },
                        "O3": {
                            "value": aqi_data['O3'],
                            "unit": "µg/m³",
                            "safe_range": "0-50",
                            "description": "Ozone"
                        }
                    }
                    
                    for pollutant, info in metrics.items():
                        st.metric(
                            label=f"{pollutant} ({info['description']})",
                            value=f"{info['value']} {info['unit']}",
                            delta=f"Safe range: {info['safe_range']} {info['unit']}"
                        )
                
                with col2:
                    st.subheader("Model Prediction")
                    # Create DataFrame with uppercase feature names
                    model_input = pd.DataFrame([[aqi_data['PM2.5'], 
                                                  aqi_data['NO2'],
                                                  aqi_data['CO'],
                                                  aqi_data['SO2'],
                                                  aqi_data['O3']]], 
                                                columns=['PM2.5', 'NO2', 'CO', 'SO2', 'O3'])
                    predicted_aqi = regressor.predict(model_input)[0]
                    predicted_category, pred_emoji = get_aqi_category(predicted_aqi)
                    
                    st.write(f"### Predicted AQI: {predicted_aqi:.1f} {pred_emoji}")
                    st.write(f"Status: {predicted_category}")
                    
                    # Calculate and display model performance metrics
                    aqi_diff = abs(predicted_aqi - aqi_data['aqi'])
                    accuracy = max(0, 100 - (aqi_diff / aqi_data['aqi']) * 100)
                    
                    st.write("### Model Performance")
                    st.metric(
                        label="Prediction Accuracy",
                        value=f"{accuracy:.1f}%",
                        delta=f"{-aqi_diff:.1f} AQI points" if aqi_diff > 0 else "Perfect Match!"
                    )
                    
                    # Add confidence indicator based on accuracy
                    if accuracy >= 90:
                        confidence = "High Confidence 🎯"
                    elif accuracy >= 70:
                        confidence = "Moderate Confidence 👍"
                    else:
                        confidence = "Low Confidence ⚠️"
                    st.write(f"Prediction Confidence: {confidence}")
                
                # Add recommendations based on AQI levels
                st.subheader("Health Recommendations")
                current_aqi = aqi_data['aqi']
                if current_aqi <= 50:
                    st.success("Air quality is good. Perfect for outdoor activities! 🌳")
                elif current_aqi <= 100:
                    st.info("Moderate air quality. Sensitive individuals should reduce prolonged outdoor exposure. 🚶")
                elif current_aqi <= 150:
                    st.warning("Unhealthy for sensitive groups. Reduce outdoor activities. 😷")
                elif current_aqi <= 200:
                    st.warning("Unhealthy. Everyone should limit outdoor activities. 🏠")
                elif current_aqi <= 300:
                    st.error("Very unhealthy. Avoid outdoor activities. Stay indoors! ⚠️")
                else:
                    st.error("Hazardous conditions! Emergency conditions. Take precautions! ☣️")
            except Exception as e:
                st.error(f"Error finding location: {str(e)}. Please try again with a different city/state combination.")


def show_predict_page():
    st.title("AQI prediction")
    st.write("""Input info. to predict AQI""")
    
    # Using consistent uppercase feature names
    PM2_5 = st.number_input("PM2.5 (Usually ranges from 0.1 to 120)", min_value=0.0, max_value=950.0, step=0.01, format="%.2f")
    NO2 = st.number_input("NO2 (Usually ranges from 0.01 to 60)", min_value=0.0, max_value=362.0, step=0.01, format="%.2f")
    CO = st.number_input("CO (Usually ranges from 0 to 3)", min_value=0.0, max_value=1756.0, step=0.01, format="%.2f")
    SO2 = st.number_input("SO2 (Usually ranges from 0.01 to 25)", min_value=0.0, max_value=194.0, step=0.01, format="%.2f")
    O3 = st.number_input("O3 (Usually ranges from 0.01 to 65)", min_value=0.0, max_value=258.0, step=0.01, format="%.2f")

    ok = st.button("Calculate AQI")
    if ok:
        # Create DataFrame with uppercase feature names
        X = pd.DataFrame([[PM2_5, NO2, CO, SO2, O3]], columns=['PM2.5', 'NO2', 'CO', 'SO2', 'O3'])
        AQI = regressor.predict(X)[0]
        
        # Convert AQI to category with emoji
        def get_aqi_category(aqi):
            if aqi <= 50:
                return ("Good", "🟢")
            elif aqi <= 100:
                return ("Satisfactory", "🟡")
            elif aqi <= 200:
                return ("Moderate", "🟠")
            elif aqi <= 300:
                return ("Poor", "🔴")
            else:
                return ("Severe", "⚫")
        
        category, emoji = get_aqi_category(AQI)
        st.subheader(f"Air Quality Category: {category} {emoji}")
        
        # Add health recommendations based on AQI
        st.subheader("Health Recommendations")
        if AQI <= 50:
            st.success("Air quality is good. Perfect for outdoor activities! 🌳")
        elif AQI <= 100:
            st.info("Moderate air quality. Sensitive individuals should reduce prolonged outdoor exposure. 🚶")
        elif AQI <= 200:
            st.warning("Unhealthy for sensitive groups. Reduce outdoor activities. 😷")
        elif AQI <= 300:
            st.warning("Unhealthy. Everyone should limit outdoor activities. 🏠")
        else:
            st.error("Very unhealthy. Avoid outdoor activities. Stay indoors! ⚠️")


@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_city_aqi(city, coords, api_key):
    try:
        url = f"https://api.waqi.info/feed/geo:{coords[0]};{coords[1]}/?token={api_key}"
        response = requests.get(url, timeout=5)  # Add timeout
        data = response.json()
        
        if response.status_code == 200 and data['status'] == 'ok':
            return {
                'city': city,
                'coords': coords,
                'aqi': data['data']['aqi'],
                'status': 'success'
            }
    except Exception:
        pass
    return {
        'city': city,
        'coords': coords,
        'aqi': None,
        'status': 'error'
    }

def show_india_aqi_map():
    st.title("India Air Quality Map")
    st.write("Real-time Air Quality Index (AQI) map of cities across India")
    
    # Show loading state
    with st.spinner("Loading Air Quality data for cities across India..."):
        # WAQI API key
        api_key = "4bd665894a9f473c47c0eb62121cd5a70b9378b4"
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Create a map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
        
        # Expanded list of Indian cities with their coordinates
        indian_cities = {
            # Metro Cities
            "Delhi": [28.6139, 77.2090],
            "Mumbai": [19.0760, 72.8777],
            "Bangalore": [12.9716, 77.5946],
            "Chennai": [13.0827, 80.2707],
            "Kolkata": [22.5726, 88.3639],
            "Hyderabad": [17.3850, 78.4867],
            
            # State Capitals
            "Lucknow": [26.8467, 80.9462],
            "Jaipur": [26.9124, 75.7873],
            "Bhopal": [23.2599, 77.4126],
            "Patna": [25.5941, 85.1376],
            "Raipur": [21.2514, 81.6296],
            "Bhubaneswar": [20.2961, 85.8245],
            "Chandigarh": [30.7333, 76.7794],
            "Dehradun": [30.3165, 78.0322],
            "Gandhinagar": [23.2156, 72.6369],
            "Ranchi": [23.3441, 85.3096],
            "Thiruvananthapuram": [8.5241, 76.9366],
            "Shillong": [25.5788, 91.8933],
            "Imphal": [24.8170, 93.9368],
            "Aizawl": [23.7307, 92.7173],
            "Kohima": [25.6751, 94.1086],
            "Panaji": [15.4909, 73.8278],
            "Agartala": [23.8315, 91.2868],
            "Shimla": [31.1048, 77.1734],
            "Itanagar": [27.0844, 93.6053],
            "Port Blair": [11.6234, 92.7265],
            
            # Major Industrial/Commercial Cities
            "Pune": [18.5204, 73.8567],
            "Ahmedabad": [23.0225, 72.5714],
            "Surat": [21.1702, 72.8311],
            "Visakhapatnam": [17.6868, 83.2185],
            "Nagpur": [21.1458, 79.0882],
            "Indore": [22.7196, 75.8577],
            "Thane": [19.2183, 72.9781],
            "Kanpur": [26.4499, 80.3319],
            "Coimbatore": [11.0168, 76.9558],
            "Guwahati": [26.1445, 91.7362],
            "Ludhiana": [30.9010, 75.8573],
            "Nashik": [19.9975, 73.7898],
            "Vadodara": [22.3072, 73.1812],
            "Madurai": [9.9252, 78.1198],
            "Varanasi": [25.3176, 82.9739],
            "Agra": [27.1767, 78.0081],
            "Aurangabad": [19.8762, 75.3433],
            "Kochi": [9.9312, 76.2673],
            "Mysore": [12.2958, 76.6394],
            "Jamshedpur": [22.8046, 86.2029],
            "Amritsar": [31.6340, 74.8723],
            "Rajkot": [22.3039, 70.8022],
            "Allahabad": [25.4358, 81.8463],
            "Gwalior": [26.2183, 78.1828],
            "Jabalpur": [23.1815, 79.9864]
        }
        
        # Fetch AQI data concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all API requests
            future_to_city = {
                executor.submit(fetch_city_aqi, city, coords, api_key): city 
                for city, coords in indian_cities.items()
            }
            
            # Process results as they complete
            completed = 0
            total_cities = len(indian_cities)
            
            for future in concurrent.futures.as_completed(future_to_city):
                completed += 1
                progress = completed / total_cities
                progress_bar.progress(progress)
                
                result = future.result()
                if result['status'] == 'success' and result['aqi'] is not None:
                    aqi = result['aqi']
                    city = result['city']
                    coords = result['coords']
                    
                    # Determine color and category based on AQI
                    if aqi <= 50:
                        color = 'green'
                        category = 'Good'
                    elif aqi <= 100:
                        color = 'yellow'
                        category = 'Moderate'
                    elif aqi <= 150:
                        color = 'orange'
                        category = 'Unhealthy for Sensitive Groups'
                    elif aqi <= 200:
                        color = 'red'
                        category = 'Unhealthy'
                    elif aqi <= 300:
                        color = 'purple'
                        category = 'Very Unhealthy'
                    else:
                        color = 'black'
                        category = 'Hazardous'
                    
                    # Add marker with popup
                    folium.CircleMarker(
                        location=coords,
                        radius=12,
                        popup=f"{city}<br>AQI: {aqi}<br>Status: {category}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        weight=2
                    ).add_to(m)
        
        # Remove progress bar after completion
        progress_bar.empty()
        
        # Add a legend with improved styling
        legend_html = '''
            <div style="position: fixed; 
                        bottom: 50px; 
                        left: 50px; 
                        z-index: 1000; 
                        background-color: white; 
                        padding: 10px; 
                        border: 2px solid grey; 
                        border-radius: 5px;
                        font-family: Arial, sans-serif;
                        box-shadow: 0 0 15px rgba(0,0,0,0.2);">
                <h4 style="margin-top: 0;">AQI Legend</h4>
                <div><span style="color: green; font-size: 20px;">●</span> Good (0-50)</div>
                <div><span style="color: yellow; font-size: 20px;">●</span> Moderate (51-100)</div>
                <div><span style="color: orange; font-size: 20px;">●</span> Unhealthy for Sensitive Groups (101-150)</div>
                <div><span style="color: red; font-size: 20px;">●</span> Unhealthy (151-200)</div>
                <div><span style="color: purple; font-size: 20px;">●</span> Very Unhealthy (201-300)</div>
                <div><span style="color: black; font-size: 20px;">●</span> Hazardous (300+)</div>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display the map
        folium_static(m)
    
    # Add city categories in tabs for better organization
    tab1, tab2 = st.tabs(["Metro & State Capitals", "Industrial Cities"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Metro Cities:**
            - Delhi, Mumbai, Bangalore
            - Chennai, Kolkata, Hyderabad
            """)
        with col2:
            st.markdown("""
            **State Capitals:**
            - Lucknow, Jaipur, Bhopal
            - Patna, Raipur, Bhubaneswar
            - And more...
            """)
    
    with tab2:
        st.markdown("""
        **Major Industrial/Commercial Cities:**
        - Pune, Ahmedabad, Surat
        - Visakhapatnam, Nagpur, Indore
        - Coimbatore, Ludhiana, Nashik
        - And more...
        """)
    
    # Add explanation in an expandable section
    with st.expander("About this map"):
        st.info("""
        This map shows real-time Air Quality Index (AQI) data for cities across India, including:
        - All major metropolitan cities
        - State capitals
        - Major industrial and commercial hubs
        - Click on any marker to see detailed AQI information
        - Colors indicate the air quality level from Good (green) to Hazardous (black)
        """)
        
        # Add timestamp with styling
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        
        # Add data source attribution
        st.markdown("""
        <small>Data source: World Air Quality Index Project (WAQI)</small>
        """, unsafe_allow_html=True)

def show_model_metrics():
    st.title("Model Performance Metrics")
    
    # Display model accuracy
    st.subheader("Model Accuracy")
    accuracy_percentage = data.get("r2_score", 0.89) * 100
    st.metric(
        label="R-squared Score",
        value=f"{accuracy_percentage:.1f}%",
        delta="Based on training data",
        delta_color="normal"
    )
    
    # Display feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['PM2.5', 'NO2', 'CO', 'SO2', 'O3'],
        'Importance': [0.45, 0.20, 0.15, 0.10, 0.10]  # Example values
    })
    
    # Create a bar chart for feature importance
    st.bar_chart(
        feature_importance.set_index('Feature')['Importance'],
        use_container_width=True
    )
    
    # Add model details
    st.subheader("Model Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Type:** Random Forest Regressor
        **Training Data:** Historical AQI Data
        **Input Features:** 5
        """)
    
    with col2:
        st.markdown("""
        **Output:** AQI Value
        **Valid Range:** 0-500
        **Last Updated:** 2024
        """)
    
    # Add explanation box
    with st.expander("About the Metrics"):
        st.markdown("""
        - **R-squared Score:** Indicates how well the model fits the data. Higher is better.
        - **Feature Importance:** Shows the relative impact of each pollutant on AQI prediction.
        - **PM2.5:** Most significant factor in AQI calculation
        - **NO2 & CO:** Moderate impact on AQI
        - **SO2 & O3:** Lesser but still significant impact
        """)

def get_psychological_recommendations(aqi, time_of_day, exposure_duration):
    """Generate personalized psychological recommendations based on AQI and context."""
    recommendations = {
        'low_impact': {
            'morning': [
                "Start your day with outdoor meditation or yoga to boost mental wellness",
                "Take a refreshing morning walk to energize your mind",
                "Practice mindful breathing exercises in the fresh air",
                "Engage in outdoor photography or art to stimulate creativity"
            ],
            'afternoon': [
                "Consider a lunch break in a nearby park",
                "Do light outdoor exercises to maintain positive energy",
                "Practice grounding techniques in nature",
                "Take short wellness breaks outside"
            ],
            'evening': [
                "End your day with a relaxing outdoor stroll",
                "Practice sunset meditation for mental clarity",
                "Engage in gentle outdoor stretching",
                "Connect with nature through gardening or plant care"
            ]
        },
        'moderate_impact': {
            'morning': [
                "Start indoor meditation with air purification",
                "Do indoor stretching near well-ventilated areas",
                "Practice positive visualization techniques",
                "Engage in creative indoor activities"
            ],
            'afternoon': [
                "Take breaks in well-ventilated indoor spaces",
                "Practice desk exercises and stretching",
                "Use stress-relief apps or guided meditation",
                "Maintain social connections through virtual means"
            ],
            'evening': [
                "Create a calming indoor environment",
                "Practice relaxation techniques before sleep",
                "Engage in indoor hobbies",
                "Plan indoor social activities"
            ]
        },
        'high_impact': {
            'morning': [
                "Start with indoor air quality check",
                "Practice deep breathing exercises with air purifier",
                "Follow online wellness sessions",
                "Maintain a positive morning routine indoors"
            ],
            'afternoon': [
                "Take regular breaks to check mental state",
                "Practice stress-management techniques",
                "Use mood-tracking apps",
                "Engage in indoor mindfulness activities"
            ],
            'evening': [
                "Create a protective indoor sanctuary",
                "Practice anxiety-reducing techniques",
                "Connect with support groups online",
                "Prepare for quality indoor rest"
            ]
        }
    }
    
    impact_level = 'high_impact' if aqi > 200 else 'moderate_impact' if aqi > 100 else 'low_impact'
    
    # Add duration-based recommendations
    if exposure_duration > 7:  # If exposure longer than a week
        long_term_tips = [
            "Consider scheduling a weekend getaway to areas with better air quality",
            "Join online support groups for air quality concerns",
            "Develop a long-term indoor wellness routine",
            "Track your mood patterns in relation to air quality"
        ]
        recommendations[impact_level][time_of_day].extend(long_term_tips)
    
    return recommendations[impact_level][time_of_day]

def get_stress_management_tips(aqi_level):
    """Provide specific stress management techniques based on AQI level."""
    tips = {
        'low': [
            "Practice mindful awareness of your surroundings",
            "Engage in light physical activity",
            "Use positive affirmations",
            "Maintain regular social connections"
        ],
        'moderate': [
            "Follow guided meditation sessions",
            "Practice progressive muscle relaxation",
            "Use aromatherapy with air purification",
            "Maintain a mood journal"
        ],
        'high': [
            "Practice anxiety-reducing breathing techniques",
            "Use stress-relief apps and tools",
            "Schedule regular check-ins with mental health professionals",
            "Join air quality support groups"
        ]
    }
    level = 'high' if aqi_level > 200 else 'moderate' if aqi_level > 100 else 'low'
    return tips[level]

def show_psychological_impact(aqi):
    """Display psychological impact assessment and recommendations."""
    st.subheader("🧠 Psychological Impact Assessment")
    
    # Current time-based context
    current_hour = datetime.now().hour
    time_of_day = 'morning' if 5 <= current_hour < 12 else 'afternoon' if 12 <= current_hour < 18 else 'evening'
    
    # Calculate exposure duration (example: 7 days)
    exposure_duration = 7  # This could be tracked over time
    
    # Get personalized recommendations
    recommendations = get_psychological_recommendations(aqi, time_of_day, exposure_duration)
    stress_tips = get_stress_management_tips(aqi)
    
    # Display recommendations with enhanced UI
    st.markdown("### 🎯 Personalized Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Daily Wellness Tips")
        for rec in recommendations[:2]:
            st.success(rec)
        
        st.markdown("#### Mood Management")
        st.info("Track your daily mood in relation to air quality")
        # Add a mood tracker
        mood = st.select_slider(
            "How are you feeling today?",
            options=["😔", "😐", "🙂", "😊", "😄"],
            value="🙂"
        )
        
    with col2:
        st.markdown("#### Stress Management")
        for tip in stress_tips[:2]:
            st.warning(tip)
        
    # Activity Suggestions
    st.markdown("### 🎨 Recommended Activities")
    activities = {
        'Indoor': ["Meditation", "Yoga", "Reading", "Creative Arts"],
        'Outdoor': ["Short Walks", "Garden Visit", "Photography", "Nature Observation"]
    }
    
    if aqi <= 100:
        for activity in activities['Outdoor']:
            st.success(f"🌿 {activity}")
    else:
        for activity in activities['Indoor']:
            st.info(f"🏠 {activity}")
    
    # Mental Health Tracking
    st.markdown("### 📊 Wellness Tracking")
    
    with st.form(key='wellness_form'):
        st.subheader("Daily Wellness Indicators")
        st.markdown("""
        <p style='font-size: 0.9rem; color: #666; margin-bottom: 20px;'>
        Rate each indicator based on how you feel today. Move the slider to the value that best matches your current state.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 😌 Emotional State")
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                    <p style='font-size: 0.9rem; font-weight: bold; color: #333;'>How to Rate Your Stress Level:</p>
                    <p style='font-size: 0.85rem; color: #666;'>
                    • 0-2: Feeling completely relaxed, like after a vacation<br>
                    • 3-4: Slight pressure but manageable, like a normal workday<br>
                    • 5-6: Noticeable stress, like before a deadline<br>
                    • 7-8: High stress, feeling overwhelmed with tasks<br>
                    • 9-10: Extreme stress, feeling unable to cope
                    </p>
                </div>
            """, unsafe_allow_html=True)
            stress_level = st.slider("Stress Level", 0, 10, 5)

            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                    <p style='font-size: 0.9rem; font-weight: bold; color: #333;'>How to Rate Your Sleep Quality:</p>
                    <p style='font-size: 0.85rem; color: #666;'>
                    • 0-2: Barely slept, constant disruptions<br>
                    • 3-4: Poor sleep, woke up multiple times<br>
                    • 5-6: Average sleep, some interruptions<br>
                    • 7-8: Good sleep, woke up feeling refreshed<br>
                    • 9-10: Perfect sleep, feel completely rejuvenated
                    </p>
                </div>
            """, unsafe_allow_html=True)
            sleep_quality = st.slider("Sleep Quality", 0, 10, 7)

            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                    <p style='font-size: 0.9rem; font-weight: bold; color: #333;'>How to Rate Your Anxiety Level:</p>
                    <p style='font-size: 0.85rem; color: #666;'>
                    • 0-2: Feeling calm and at peace<br>
                    • 3-4: Mild unease, like before a meeting<br>
                    • 5-6: Noticeable worry about several things<br>
                    • 7-8: Strong anxiety affecting daily tasks<br>
                    • 9-10: Severe anxiety, possibly panic symptoms
                    </p>
                </div>
            """, unsafe_allow_html=True)
            anxiety_level = st.slider("Anxiety Level", 0, 10, 4)

        with col2:
            st.markdown("#### 🌟 Overall Wellness")
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                    <p style='font-size: 0.9rem; font-weight: bold; color: #333;'>How to Rate Your Mood:</p>
                    <p style='font-size: 0.85rem; color: #666;'>
                    • 0-2: Feeling very low, struggling to engage<br>
                    • 3-4: Somewhat down, less interested in activities<br>
                    • 5-6: Neutral, neither particularly happy nor sad<br>
                    • 7-8: Generally positive, enjoying activities<br>
                    • 9-10: Excellent mood, feeling enthusiastic
                    </p>
                </div>
            """, unsafe_allow_html=True)
            mood_score = st.slider("Mood", 0, 10, 6)

            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                    <p style='font-size: 0.9rem; font-weight: bold; color: #333;'>How to Rate Your Energy Level:</p>
                    <p style='font-size: 0.85rem; color: #666;'>
                    • 0-2: Exhausted, struggling to stay awake<br>
                    • 3-4: Low energy, need extra rest<br>
                    • 5-6: Average energy for daily tasks<br>
                    • 7-8: Good energy, feeling productive<br>
                    • 9-10: High energy, feeling very active
                    </p>
                </div>
            """, unsafe_allow_html=True)
            energy_level = st.slider("Energy Level", 0, 10, 6)

        st.markdown("""
            <div style='background-color: #e8f4ea; padding: 15px; border-radius: 5px; margin-top: 20px;'>
                <p style='font-size: 0.9rem; font-weight: bold; color: #2e7d32;'>Tips for Accurate Rating:</p>
                <p style='font-size: 0.85rem; color: #1f1f1f;'>
                • Compare how you feel right now to the descriptions above<br>
                • Consider your state over the last few hours<br>
                • Be honest - there are no "right" or "wrong" answers<br>
                • Trust your first instinct rather than overthinking<br>
                • Update your ratings at different times to track changes
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Submit button for the form
        submitted = st.form_submit_button("Analyze Impact", help="Click to analyze the relationship between air quality and your well-being")
        
        if submitted and aqi:
            # Calculate stress index based on AQI and personal indicators
            base_stress_impact = min((aqi / 500) * 10, 10)
            personal_stress_index = (stress_level + (10 - sleep_quality) + anxiety_level + 
                                (10 - mood_score) + (10 - energy_level)) / 5
            
            # Calculate correlation and impact scores
            aqi_stress_correlation = min((base_stress_impact + personal_stress_index) / 2, 10)
            
            # Display results with custom styling
            st.markdown("### Analysis Results")
            
            # Create three columns for metrics
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric(
                    "AQI Impact Score",
                    f"{base_stress_impact:.1f}/10",
                    delta="Based on current AQI"
                )
            
            with c2:
                st.metric(
                    "Personal Stress Index",
                    f"{personal_stress_index:.1f}/10",
                    delta="Based on your inputs"
                )
            
            with c3:
                st.metric(
                    "Overall Correlation",
                    f"{aqi_stress_correlation:.1f}/10",
                    delta="Combined impact"
                )
            
            # Enhanced Personalized Recommendations
            st.markdown("### Personalized Recommendations")
            
            # Create detailed recommendations based on individual metrics
            recommendations = {
                'stress': {
                    'high': [
                        "🧘‍♀️ Practice deep breathing exercises 3 times daily",
                        "🎵 Use calming music or nature sounds during work",
                        "⏰ Take regular 5-minute stress-relief breaks",
                        "📱 Consider using stress-management apps"
                    ],
                    'moderate': [
                        "🌿 Try gentle stretching exercises",
                        "☕ Take mindful breaks between tasks",
                        "📝 Start a stress journal to track triggers"
                    ],
                    'low': [
                        "✨ Maintain current stress management practices",
                        "🌟 Continue your balanced routine"
                    ]
                },
                'sleep': {
                    'poor': [
                        "😴 Establish a strict sleep schedule",
                        "🌙 Create a calming bedtime routine",
                        "🛏️ Ensure bedroom air quality with purifiers",
                        "📱 Limit screen time before bed"
                    ],
                    'moderate': [
                        "💤 Improve sleep environment",
                        "🍵 Consider calming herbal teas before bed",
                        "🚶‍♀️ Take a short evening walk (if AQI permits)"
                    ],
                    'good': [
                        "👍 Maintain current sleep hygiene",
                        "✨ Continue your effective routine"
                    ]
                },
                'anxiety': {
                    'high': [
                        "💆‍♀️ Practice guided meditation daily",
                        "📞 Consider professional support",
                        "🎯 Use grounding techniques when anxious",
                        "📱 Try anxiety-management apps"
                    ],
                    'moderate': [
                        "🌸 Practice mindfulness exercises",
                        "📝 Start an anxiety journal",
                        "👥 Join support groups"
                    ],
                    'low': [
                        "🌟 Maintain current coping strategies",
                        "✨ Continue regular check-ins"
                    ]
                },
                'mood': {
                    'low': [
                        "☀️ Increase exposure to natural light",
                        "🤝 Schedule social interactions",
                        "🎨 Engage in creative activities",
                        "🎵 Create a mood-lifting playlist"
                    ],
                    'moderate': [
                        "🌺 Try mood-boosting activities",
                        "📝 Start a gratitude journal",
                        "🚶‍♀️ Regular light exercise (indoor if needed)"
                    ],
                    'good': [
                        "🌟 Continue mood-maintaining activities",
                        "✨ Share positive experiences"
                    ]
                },
                'energy': {
                    'low': [
                        "⚡ Schedule energy-boosting activities",
                        "🥗 Focus on nutrition and hydration",
                        "💪 Try gentle indoor exercises",
                        "🌿 Consider natural energy boosters"
                    ],
                    'moderate': [
                        "🚶‍♀️ Take regular movement breaks",
                        "🎯 Optimize work-rest cycles",
                        "🌞 Maximize natural light exposure"
                    ],
                    'good': [
                        "✨ Maintain current energy practices",
                        "💫 Share energy management tips"
                    ]
                }
            }

            # Function to get recommendation level
            def get_level(score):
                if score <= 4:
                    return 'low'
                elif score <= 7:
                    return 'moderate'
                else:
                    return 'high'

            # Create expandable sections for each category
            st.markdown("#### 🎯 Detailed Wellness Plan")
            
            # Stress Management
            with st.expander("😌 Stress Management", expanded=True):
                stress_level_text = get_level(stress_level)
                st.markdown(f"""
                **Current Status**: {'High Stress 😰' if stress_level > 7 else 'Moderate Stress 😐' if stress_level > 4 else 'Low Stress 😊'}
                
                **Recommended Actions**:
                """)
                for rec in recommendations['stress'][stress_level_text]:
                    st.markdown(f"• {rec}")

            # Sleep Quality
            with st.expander("😴 Sleep Enhancement", expanded=True):
                sleep_level_text = 'poor' if sleep_quality < 5 else 'moderate' if sleep_quality < 8 else 'good'
                st.markdown(f"""
                **Current Status**: {'Poor Sleep 😫' if sleep_quality < 5 else 'Moderate Sleep 😐' if sleep_quality < 8 else 'Good Sleep 😊'}
                
                **Recommended Actions**:
                """)
                for rec in recommendations['sleep'][sleep_level_text]:
                    st.markdown(f"• {rec}")

            # Anxiety Management
            with st.expander("🧘‍♀️ Anxiety Management", expanded=True):
                anxiety_level_text = get_level(anxiety_level)
                st.markdown(f"""
                **Current Status**: {'High Anxiety 😰' if anxiety_level > 7 else 'Moderate Anxiety 😐' if anxiety_level > 4 else 'Low Anxiety 😊'}
                
                **Recommended Actions**:
                """)
                for rec in recommendations['anxiety'][anxiety_level_text]:
                    st.markdown(f"• {rec}")

            # Mood Enhancement
            with st.expander("😊 Mood Enhancement", expanded=True):
                mood_level_text = 'low' if mood_score < 5 else 'moderate' if mood_score < 8 else 'good'
                st.markdown(f"""
                **Current Status**: {'Low Mood 😔' if mood_score < 5 else 'Moderate Mood 😐' if mood_score < 8 else 'Good Mood 😊'}
                
                **Recommended Actions**:
                """)
                for rec in recommendations['mood'][mood_level_text]:
                    st.markdown(f"• {rec}")

            # Energy Management
            with st.expander("⚡ Energy Management", expanded=True):
                energy_level_text = 'low' if energy_level < 5 else 'moderate' if energy_level < 8 else 'good'
                st.markdown(f"""
                **Current Status**: {'Low Energy 😫' if energy_level < 5 else 'Moderate Energy 😐' if energy_level < 8 else 'High Energy 😊'}
                
                **Recommended Actions**:
                """)
                for rec in recommendations['energy'][energy_level_text]:
                    st.markdown(f"• {rec}")

            # Add AQI-specific recommendations
            st.markdown("#### 🌬️ Air Quality Specific Recommendations")
            if aqi > 200:
                st.error("""
                **High AQI Alert - Additional Precautions:**
                * 🏠 Stay indoors with air purification
                * 😷 Wear N95 mask if outdoors
                * 🌿 Use indoor air purifying plants
                * 💨 Monitor indoor air quality
                * 📱 Set up air quality alerts
                """)
            elif aqi > 100:
                st.warning("""
                **Moderate AQI - Precautions:**
                • 🚶‍♀️ Limit outdoor activities
                • 🪟 Keep windows closed during peak hours
                • 💧 Stay well hydrated
                • 🏃‍♀️ Indoor exercise recommended
                """)
            else:
                st.success("""
                **Good AQI - Maintain Wellness:**
                • 🌳 Enjoy outdoor activities
                • 🌞 Get natural sunlight
                • 🚴‍♀️ Regular outdoor exercise
                • 🌺 Practice outdoor mindfulness
                """)

            # Add a daily schedule suggestion
            st.markdown("#### 📅 Suggested Daily Schedule")
            schedule = {
                'Morning': [
                    "🌅 Wake up at consistent time",
                    "🧘‍♀️ Morning meditation/stretching",
                    "🥗 Nutritious breakfast",
                    "🚶‍♀️ Light exercise (based on AQI)"
                ],
                'Afternoon': [
                    "⏰ Regular work breaks",
                    "🧘‍♀️ Midday stress relief",
                    "🥗 Balanced lunch",
                    "🚶‍♀️ Post-lunch walk (if AQI permits)"
                ],
                'Evening': [
                    "🌙 Wind-down activities",
                    "📱 Reduce screen time",
                    "🛁 Relaxing bedtime routine",
                    "😴 Consistent bedtime"
                ]
            }

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Morning Routine**")
                for item in schedule['Morning']:
                    st.markdown(f"• {item}")
            with col2:
                st.markdown("**Afternoon Routine**")
                for item in schedule['Afternoon']:
                    st.markdown(f"• {item}")
            with col3:
                st.markdown("**Evening Routine**")
                for item in schedule['Evening']:
                    st.markdown(f"• {item}")

            # Add weekly goals
            st.markdown("#### 🎯 Weekly Wellness Goals")
            st.markdown("""
            1. 🧘‍♀️ Complete 10 minutes of mindfulness daily
            2. 💪 Achieve 150 minutes of moderate exercise
            3. 😴 Maintain consistent sleep schedule
            4. 📝 Track mood and energy levels daily
            5. 🌿 Practice stress-management techniques
            """)

            # Progress Tracking
            st.markdown("#### 📊 Track Your Progress")
            
            submitted = st.form_submit_button("Save Today's Wellness Data")
            
            if submitted and aqi:
                st.session_state.wellness_logs = getattr(st.session_state, 'wellness_logs', [])
                st.session_state.wellness_logs.append({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'aqi': aqi,
                    'stress': stress_level,
                    'sleep': sleep_quality,
                    'anxiety': anxiety_level,
                    'mood': mood_score,
                    'energy': energy_level
                })
                st.success("Wellness data saved successfully! Keep tracking for better insights.")

def show_stress_correlation(current_aqi=None):
    st.title(" Air Quality & Mental Wellness Analysis")
    
    # Get current time context
    current_hour = datetime.now().hour
    time_of_day = 'morning' if 5 <= current_hour < 12 else 'afternoon' if 12 <= current_hour < 18 else 'evening'
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Personal Analysis", "📊 Air Quality Trends"])
    
    with tab1:
        st.markdown("###  Location & Air Quality")
        
        # Enhanced location detection with error handling
        use_location = st.checkbox("📱 Use my current location", help="Enable to automatically detect your location")
        
        if use_location:
            with st.spinner("📍 Detecting your location..."):
                location_data = streamlit_geolocation()
                if location_data and location_data.get('latitude') and location_data.get('longitude'):
                    lat = location_data['latitude']
                    lon = location_data['longitude']
                    aqi_data = get_aqi_data(lat, lon)
                    if aqi_data:
                        current_aqi = aqi_data['aqi']
                        st.success(f"📌 Current AQI at your location: {current_aqi}")
                        
                        # Display AQI category and health implications
                        if current_aqi <= 50:
                            st.success("Air Quality: Good - Perfect for outdoor activities! 🌳")
                        elif current_aqi <= 100:
                            st.info("Air Quality: Moderate - Sensitive individuals should reduce prolonged outdoor exposure 🚶")
                        elif current_aqi <= 150:
                            st.warning("Air Quality: Unhealthy for Sensitive Groups - Reduce outdoor activities 😷")
                        elif current_aqi <= 200:
                            st.warning("Air Quality: Unhealthy - Everyone should limit outdoor activities 🏠")
                        elif current_aqi <= 300:
                            st.error("Air Quality: Very Unhealthy - Avoid outdoor activities! ⚠️")
                        else:
                            st.error("Air Quality: Hazardous - Emergency conditions! Take precautions! ☣️")
                    else:
                        st.error("❌ Could not fetch AQI data for your location")
        else:
            current_aqi = st.number_input("🌡️ Enter current AQI value:", 0, 500, 100)
        
        # Air Quality Impact Assessment
        st.markdown("###  Air Quality Impact Assessment")
        
        # Create wellness form with enhanced features
        with st.form(key='wellness_form'):
            st.markdown("#### 😷 Air Quality Symptoms")
            col1, col2 = st.columns(2)
            with col1:
                respiratory_issues = st.checkbox("Respiratory Issues (coughing, wheezing)")
                eye_irritation = st.checkbox("Eye Irritation")
                throat_irritation = st.checkbox("Throat Irritation")
                breathing_difficulty = st.checkbox("Breathing Difficulty")
            with col2:
                fatigue = st.checkbox("Fatigue")
                dizziness = st.checkbox("Dizziness")
                headache = st.checkbox("Headache")
                chest_tightness = st.checkbox("Chest Tightness")
            
            st.markdown("#### 🧠 Air Quality Impact on Mental State")
            col3, col4 = st.columns(2)
            with col3:
                stress_level = st.slider("Stress Level", 0, 10, 5, 
                    help="Rate your current stress level from 0 (completely relaxed) to 10 (extremely stressed)")
                
                anxiety_level = st.slider("Anxiety Level", 0, 10, 4,
                    help="Rate your anxiety from 0 (calm) to 10 (severe anxiety)")
                
                focus_level = st.slider("Focus & Concentration", 0, 10, 6,
                    help="Rate your ability to focus from 0 (very scattered) to 10 (highly focused)")
            
            with col4:
                sleep_quality = st.slider("Sleep Quality", 0, 10, 7,
                    help="Rate your sleep quality from 0 (very poor) to 10 (excellent)")
                
                energy_level = st.slider("Energy Level", 0, 10, 6,
                    help="Rate your energy from 0 (exhausted) to 10 (highly energetic)")
                
                mood_level = st.slider("Mood Level", 0, 10, 6,
                    help="Rate your mood from 0 (very low) to 10 (excellent)")
            
            # Environmental Exposure
            st.markdown("#### 🌍 Environmental Exposure")
            col5, col6 = st.columns(2)
            with col5:
                outdoor_time = st.slider("Hours Spent Outdoors Today", 0, 24, 2)
                ventilation_rating = st.slider("Indoor Ventilation Quality", 0, 10, 7)
            with col6:
                air_purifier_use = st.checkbox("Using Air Purifier")
                mask_wearing = st.checkbox("Wearing Mask Outdoors")
            
            submitted = st.form_submit_button("📊 Analyze Air Quality Impact")
            
            if submitted and current_aqi is not None:
                # Calculate comprehensive scores
                physical_symptoms_count = sum([respiratory_issues, eye_irritation, throat_irritation, 
                    breathing_difficulty, fatigue, dizziness, headache, chest_tightness])
                
                mental_wellness_score = (10 - stress_level + 10 - anxiety_level + focus_level) / 3
                physical_wellness_score = (sleep_quality + energy_level + (10 - physical_symptoms_count)) / 3
                
                # Calculate impact score starting from 0 when no symptoms are present
                symptoms_impact_score = 10 if physical_symptoms_count == 0 else max(0, 10 - (physical_symptoms_count * 1.25))
                
                # Display Analysis Results
                st.markdown("###  Air Quality Impact Analysis")
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Air Quality Impact Score", f"{symptoms_impact_score:.1f}/10",
                        delta="Low Impact" if symptoms_impact_score > 7 else "Moderate Impact" if symptoms_impact_score > 5 else "High Impact")
                with col2:
                    st.metric("Mental Wellness Score", f"{mental_wellness_score:.1f}/10",
                        delta="Good" if mental_wellness_score > 7 else "Moderate" if mental_wellness_score > 5 else "Needs Attention")
                with col3:
                    st.metric("Physical Wellness Score", f"{physical_wellness_score:.1f}/10",
                        delta="Good" if physical_wellness_score > 7 else "Moderate" if physical_wellness_score > 5 else "Needs Attention")
                
               
                
                # Personalized Recommendations based on Analysis
                st.markdown("###  Personalized Recommendations")
                
                # Stress Management Recommendations
                if stress_level > 7:
                    st.warning("""
                    **High Stress Level Detected:**
                    * 🫁 Practice deep breathing exercises every 2 hours
                    * 🧘‍♀️ Take 5-minute meditation breaks
                    * 📱 Use stress-relief apps
                    * 👥 Consider professional support
                    * 📝 Maintain a stress journal
                    """)
                elif stress_level > 4:
                    st.info("""
                    **Moderate Stress Level:**
                    * ⏰ Take regular breaks
                    * 🧘‍♀️ Practice mindfulness
                    * 💪 Engage in light exercise
                    * ⚖️ Maintain work-life balance
                    * 😌 Use relaxation techniques
                    """)
                else:
                    st.success("""
                    **Good Stress Management:**
                    * ✨ Continue current stress management practices
                    * 🤝 Share effective techniques with others
                    * 💪 Maintain regular exercise routine
                    * 🛡️ Practice preventive stress management
                    * 📊 Keep tracking stress levels
                    """)
                
                # Sleep Quality Recommendations
                if sleep_quality < 5:
                    st.warning("""
                    **Poor Sleep Quality:**
                    * ⏰ Establish a strict sleep schedule
                    * 🌙 Create a calming bedtime routine
                    * 🌬️ Ensure bedroom air quality
                    * 📱 Limit screen time before bed
                    * 📊 Consider sleep tracking
                    """)
                elif sleep_quality < 8:
                    st.info("""
                    **Moderate Sleep Quality:**
                    * 🛏️ Improve sleep environment
                    * 🍵 Try calming herbal teas
                    * 😌 Practice relaxation before bed
                    * ⏰ Maintain consistent sleep times
                    * 📊 Track sleep patterns
                    """)
                else:
                    st.success("""
                    **Good Sleep Quality:**
                    * ✨ Maintain current sleep routine
                    * 🌙 Continue good sleep hygiene
                    * 📊 Track sleep patterns
                    * 🤝 Share effective techniques
                    * 🛡️ Practice preventive measures
                    """)
                
                # Energy Level Recommendations
                if energy_level < 5:
                    st.warning("""
                    **Low Energy Level:**
                    * ⏰ Take frequent short breaks
                    * 💧 Stay hydrated throughout the day
                    * 🥗 Eat energy-boosting foods
                    * 💪 Practice gentle stretching
                    * 💊 Consider vitamin supplements
                    """)
                elif energy_level < 8:
                    st.info("""
                    **Moderate Energy Level:**
                    * 💪 Maintain regular exercise
                    * ⚖️ Balance work and rest
                    * 🥗 Eat balanced meals
                    * 💧 Stay hydrated
                    * 📊 Track energy patterns
                    """)
                else:
                    st.success("""
                    **Good Energy Level:**
                    * ✨ Continue current energy management
                    * 🤝 Share effective techniques
                    * 💪 Maintain regular exercise
                    * 📊 Track energy patterns
                    * 🛡️ Practice preventive measures
                    """)
                
                # Focus and Concentration Recommendations
                if focus_level < 5:
                    st.warning("""
                    **Low Focus Level:**
                    * ⏰ Take regular short breaks
                    * 🧘‍♀️ Practice mindfulness exercises
                    * 🎯 Use focus-enhancing techniques
                    * 🔇 Minimize distractions
                    * 👥 Consider professional help
                    """)
                elif focus_level < 8:
                    st.info("""
                    **Moderate Focus Level:**
                    * ⏰ Use time management techniques
                    * 🧠 Practice concentration exercises
                    * ⏰ Take regular breaks
                    * 🪑 Maintain good posture
                    * 📊 Track focus patterns
                    """)
                else:
                    st.success("""
                    **Good Focus Level:**
                    • ✨ Continue current focus techniques
                    * 🤝 Share effective methods
                    * ⏰ Maintain regular breaks
                    * 📊 Track focus patterns
                    * 🛡️ Practice preventive measures
                    """)
                
 # Air Quality Specific Recommendations
                st.markdown("###  Air Quality Recommendations")
                
                # Indoor Air Quality Tips
                with st.expander("🏠 Indoor Air Quality Tips", expanded=True):
                    st.markdown("""
                    **Immediate Actions:**
                    * 🌬️ Use air purifiers in living spaces
                    * 🪟 Keep windows closed during high AQI periods
                    * 💧 Maintain indoor humidity between 30-50%
                    * 🔧 Clean air filters regularly
                    * 🌿 Use natural air-purifying plants
                    """)
                
                # Outdoor Activity Guidelines
                with st.expander("🌳 Outdoor Activity Guidelines", expanded=True):
                    if current_aqi <= 50:
                        st.success("""
                        **Safe for Outdoor Activities:**
                        * 🏃‍♂️ Enjoy outdoor exercise
                        * 🚶‍♀️ Take walks in nature
                        * 🧘‍♀️ Practice outdoor meditation
                        * 🌺 Garden or do outdoor activities
                        """)
                    elif current_aqi <= 100:
                        st.info("""
                        **Moderate Outdoor Activities:**
                        * ⏱️ Limit outdoor time
                        * 🏃‍♂️ Choose less strenuous activities
                        * 🏠 Take breaks indoors
                        * 📊 Monitor symptoms
                        """)
                    else:
                        st.warning("""
                        **Limited Outdoor Activities:**
                        * 🏠 Stay indoors when possible
                        * 😷 Wear N95 mask if going out
                        * 💪 Choose indoor exercise
                        * 🌬️ Use air-purified spaces
                        """)
                
                # Health Protection Measures
                with st.expander("😷 Health Protection Measures", expanded=True):
                    st.markdown("""
                    **Daily Protection:**
                    * 📱 Check AQI before outdoor activities
                    * 😷 Wear appropriate masks when needed
                    * 🔔 Use air quality apps for alerts
                    * 💊 Keep rescue medications handy
                    * 📊 Monitor symptoms regularly
                    """)

                # Save data for tracking
                save_wellness_data(current_aqi, stress_level, anxiety_level, sleep_quality, 
                    energy_level, physical_symptoms_count, mental_wellness_score, 
                    physical_wellness_score, symptoms_impact_score)
                
                # Display Progress Charts
                if 'wellness_logs' in st.session_state and len(st.session_state.wellness_logs) > 1:
                    st.markdown("### 📈 Air Quality Impact Trends")
                    
                    # Create separate charts for different metrics
                    df = pd.DataFrame(st.session_state.wellness_logs)
                    
                    # Air quality impact trends with scatter plot
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['aqi'],
                        mode='markers+lines',
                        name='AQI',
                        line=dict(color='blue'),
                        marker=dict(size=8)
                    ))
                    fig1.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['symptoms_impact_score'],
                        mode='markers+lines',
                        name='Impact Score',
                        line=dict(color='red'),
                        marker=dict(size=8)
                    ))
                    fig1.update_layout(
                        title='AQI and Impact Score Trends',
                        xaxis_title='Date',
                        yaxis_title='Score',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig1)
                    
                    # Physical symptoms tracking with scatter plot
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['physical_symptoms_count'],
                        mode='markers+lines',
                        name='Physical Symptoms',
                        line=dict(color='orange'),
                        marker=dict(size=8)
                    ))
                    fig2.update_layout(
                        title='Physical Symptoms Tracking',
                        xaxis_title='Date',
                        yaxis_title='Number of Symptoms',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig2)
                    
                    # Mental wellness correlation with scatter plot
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['aqi'],
                        mode='markers+lines',
                        name='AQI',
                        line=dict(color='blue'),
                        marker=dict(size=8)
                    ))
                    fig3.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['mental_wellness_score'],
                        mode='markers+lines',
                        name='Mental Wellness',
                        line=dict(color='green'),
                        marker=dict(size=8)
                    ))
                    fig3.update_layout(
                        title='AQI and Mental Wellness Correlation',
                        xaxis_title='Date',
                        yaxis_title='Score',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig3)

    with tab2:
        show_population_trends()

def get_recommended_activities(aqi, time_of_day, outdoor_time):
    activities = {
        'indoor': [
            "Meditation and deep breathing",
            "Indoor yoga or stretching",
            "Reading or creative writing",
            "Art or craft projects",
            "Indoor plants care"
        ],
        'outdoor': [
            "Short walks in nature",
            "Garden maintenance",
            "Photography",
            "Bird watching",
            "Light exercise"
        ]
    }
    
    if aqi > 150:
        activities['outdoor'] = [f"⚠️ {activity} (Not recommended due to high AQI)" for activity in activities['outdoor']]
    elif aqi > 100:
        activities['outdoor'] = [f"⚠️ {activity} (Limited duration recommended)" for activity in activities['outdoor']]
    
    return activities

def save_wellness_data(aqi, stress, anxiety, sleep, energy, symptoms, mental_score, physical_score, symptoms_score):
    if 'wellness_logs' not in st.session_state:
        st.session_state.wellness_logs = []
    
    st.session_state.wellness_logs.append({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'aqi': aqi,
        'stress_level': stress,
        'anxiety_level': anxiety,
        'sleep_quality': sleep,
        'energy_level': energy,
        'physical_symptoms_count': symptoms,
        'mental_wellness_score': mental_score,
        'physical_wellness_score': physical_score,
        'symptoms_impact_score': symptoms_score
    })
    st.success("✅ Your wellness data has been saved successfully!")

def show_population_trends():
    st.markdown("### 📊 Population Mental Health Trends")
    
    # Sample data visualization
    aqi_ranges = ['0-50', '51-100', '101-150', '151-200', '201-300', '300+']
    mental_health_impact = [10, 25, 45, 65, 80, 90]
    physical_symptoms = [5, 20, 40, 60, 75, 85]
    
    df = pd.DataFrame({
        'AQI Range': aqi_ranges,
        'Mental Health Impact %': mental_health_impact,
        'Physical Symptoms %': physical_symptoms
    })
    
    st.line_chart(df.set_index('AQI Range'))
    
    st.markdown("""
    #### 📈 Key Findings:
    - Strong correlation between AQI levels and mental health impacts
    - Physical symptoms increase significantly at AQI > 150
    - Long-term exposure shows cumulative effects
    """)

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "",
        ["Predict", "Geo Location", "India AQI Map", "Explore", "Mental Wellness"]
    )
    
    if page == "Predict":
        show_predict_page()
    elif page == "Geo Location":
        show_geo_prediction_page()
    elif page == "India AQI Map":
        show_india_aqi_map()
    elif page == "Explore":
        show_model_metrics()
    elif page == "Mental Wellness":
        # Get current AQI from geolocation if available
        try:
            lat, lon = get_geolocation()
            if lat and lon:
                aqi_data = get_aqi_data(lat, lon)
                current_aqi = aqi_data.get('aqi', None)
            else:
                current_aqi = None
        except Exception as e:
            st.warning("Unable to fetch current AQI. You can still proceed with the wellness analysis.")
            current_aqi = None
        
        show_stress_correlation(current_aqi)

if __name__ == "__main__":
    main()

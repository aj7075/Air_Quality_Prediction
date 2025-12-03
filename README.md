# Air Quality Index Prediction 🌍

## Overview
This repository contains a comprehensive Air Quality Index (AQI) prediction system that helps users determine air quality based on various pollutant parameters. The application provides both manual input prediction and real-time geolocation-based air quality monitoring.

## 🎯 Features
- **Manual Prediction**: Input pollutant levels manually to predict AQI
- **Geolocation-based Prediction**: Get real-time AQI predictions based on your current location
- **Interactive Visualization**: Explore relationships between different pollutants through charts and graphs
- **Health Recommendations**: Receive health advice based on the predicted AQI level
- **Model Performance Metrics**: View the model's accuracy and reliability

## 🌟 Novelty Features
### 1. Real-time Geolocation Integration
- **Automatic Location Detection**: Uses browser's geolocation API to detect user's current location
- **WAQI API Integration**: Fetches real-time air quality data from the World Air Quality Index Project
- **Dynamic Updates**: Continuously updates AQI predictions based on location changes
- **Global Coverage**: Works with monitoring stations worldwide

### 2. Smart Health Recommendations
- **Contextual Advice**: Provides health recommendations based on:
  - Current AQI level
  - Time of day
  - User location
- **Color-coded Alerts**: Visual indicators for different air quality levels
- **Activity Recommendations**: Suggests outdoor activity modifications
- **Health Risk Indicators**: Shows specific risks for sensitive groups

### 3. Advanced Visualization System
- **Interactive Charts**: Dynamic visualization of pollutant relationships
- **Trend Analysis**: Historical AQI data visualization
- **Pollutant Comparison**: Side-by-side comparison of different pollutants
- **Geospatial Visualization**: Map-based representation of AQI levels

### 4. Hybrid Prediction System
- **Multi-model Approach**: Combines multiple ML models for better accuracy
- **Real-time Validation**: Cross-validates predictions with actual sensor data
- **Confidence Metrics**: Displays prediction confidence levels
- **Adaptive Learning**: Improves predictions based on historical accuracy

### 5. User Experience Enhancements
- **Responsive Design**: Works seamlessly across devices
- **Intuitive Interface**: Easy-to-understand pollution metrics
- **Quick Share**: Share AQI reports via social media or email
- **Customizable Alerts**: Set personal AQI thresholds for notifications

### 6. Psychological Impact Assessment
- **Personalized Mental Wellness Recommendations**:
  - Time-of-day based suggestions (Morning/Afternoon/Evening)
  - Exposure duration-aware recommendations
  - Adaptive activity suggestions based on AQI levels
  - Indoor/Outdoor activity balance recommendations

- **Interactive Mood Tracking**:
  - Daily mood logging with emoji-based interface
  - Correlation analysis between AQI and mood patterns
  - Personal wellness journal integration
  - Historical mood data visualization

- **Stress Management System**:
  - Customized stress-reduction techniques
  - AQI-level specific coping strategies
  - Guided meditation and relaxation exercises
  - Progressive stress monitoring

- **Wellness Activity Planner**:
  - Smart scheduling based on air quality forecasts
  - Alternative indoor activity suggestions
  - Nature connection opportunities
  - Social engagement recommendations

- **Support Network Integration**:
  - Access to air quality support groups
  - Community connection features
  - Professional mental health resources
  - Regular wellness check-ins

## 🔍 Key Parameters
The system considers the following pollutants for AQI prediction:
- PM2.5 (Fine particulate matter)
- NO2 (Nitrogen Dioxide)
- SO2 (Sulfur Dioxide)
- CO (Carbon Monoxide)
- O3 (Ozone)

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Models**: 
  - XGBoost (Primary model)
  - Random Forest
  - Decision Tree
  - Linear Regression
  - Lasso Regression
  - Artificial Neural Network
- **APIs**: WAQI API for real-time air quality data

## 📊 Model Performance
- Model Accuracy: 89%
- Evaluation Metrics: R-squared score, MSE, RMSE
- Cross-validation implemented for robust performance

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/SahilAli987/AirQualityPrediction.git
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📚 Dependencies
- Python 3.5+
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- requests
- matplotlib
- seaborn

## 📂 Project Structure
```
├── Data/
│   ├── city_hour.csv        # Raw dataset
│   └── final_data.csv       # Cleaned dataset
├── models/
│   ├── Linear_Regression.ipynb
│   ├── Decision_Tree.ipynb
│   ├── Random_Forest.ipynb
│   └── XGBoost.ipynb
├── app.py                   # Main application file
├── prediction.py           # Prediction logic
├── explore_page.py        # Data exploration page
└── requirements.txt       # Project dependencies
```

## 🌟 Recent Updates
- Added geolocation-based prediction feature
- Implemented health recommendations based on AQI levels
- Fixed feature name consistency across the application
- Enhanced UI with better visualization and user feedback
- Added model accuracy display

## 👥 Contributors
- MD SAHIL ALI
- ARYAN JAIN

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/SahilAli987/AirQualityPrediction/issues).

## 📧 Contact
For any queries or suggestions, please reach out to the contributors.

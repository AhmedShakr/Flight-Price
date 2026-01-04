import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set Page Config
st.set_page_config(page_title="Flight Price System", layout="wide", page_icon="✈️")

# --- 1. Helper Functions ---
@st.cache_data
def load_data():
    try:
        x_train = pd.read_parquet('x_train.parquet')
        x_test = pd.read_parquet('x_test.parquet')
        y_test = pd.read_parquet('y_test.parquet')
        
        # CLEANING: Strip whitespace
        for col in ['Airline', 'Source', 'Destination', 'Route']:
            if col in x_train.columns:
                x_train[col] = x_train[col].astype(str).str.strip()
                
        return x_train, x_test, y_test
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None

@st.cache_resource
def load_model():
    try:
        return joblib.load('AirFlights_HistBoost_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_day_name(day_num, month_name):
    try:
        date_str = f"{int(day_num)}-{month_name}-2019"
        return pd.to_datetime(date_str, format="%d-%B-%Y").day_name()
    except:
        return "Monday"

def get_day_quarter(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# --- 2. EXHAUSTIVE AIRPORT CODE MAPPING ---
CITY_TO_CODE = {
    'Banglore': 'BLR',
    'Bangalore': 'BLR',
    'Delhi': 'DEL',
    'New Delhi': 'DEL',
    'Kolkata': 'CCU',
    'Calcutta': 'CCU',
    'Hyderabad': 'HYD',
    'Chennai': 'MAA',
    'Madras': 'MAA',
    'Mumbai': 'BOM',
    'Bombay': 'BOM',
    'Cochin': 'COK',
    'Kochi': 'COK',
    'Pune': 'PNQ',
    'Goa': 'GOI',
    'Jaipur': 'JAI',
    'Lucknow': 'LKO',
    'Patna': 'PAT',
    'Varanasi': 'VNS',
    'Bhubaneswar': 'BBI',
    'Nagpur': 'NAG',
    'Trivandrum': 'TRV'
}

def get_code(city):
    clean_city = city.strip()
    return CITY_TO_CODE.get(clean_city, clean_city[:3].upper())

def is_route_valid(route_str, source_code, dest_code):
    if pd.isna(route_str) or route_str == 'nan':
        return False
    route_upper = route_str.upper()
    parts = route_upper.replace("→", " ").replace("->", " ").split()
    if not parts:
        return False
    first_stop = parts[0]
    return (first_stop == source_code) and (dest_code in parts)

# Load Data
x_train, x_test, y_test = load_data()
model = load_model()

if x_train is None or model is None:
    st.stop()

if isinstance(y_test, pd.DataFrame):
    y_test_series = y_test.iloc[:, 0]
else:
    y_test_series = y_test

# Build Lookup
route_lookup = {}
if 'Route' in x_train.columns and 'Total_Stops' in x_train.columns:
    temp = x_train[['Route', 'Total_Stops']].drop_duplicates(subset=['Route'])
    route_lookup = temp.set_index('Route')['Total_Stops'].to_dict()

# --- 3. App Layout ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["💰 Price Prediction", "📊 Model Evaluation"])

if page == "💰 Price Prediction":
    st.title("✈️ Flight Price Prediction")
    st.markdown("### Enter Flight Details")
    
    # REMOVED st.form HERE so inputs update instantly!
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Flight Info")
        
        # Source (Updates instantly now)
        source = st.selectbox("Source", sorted(x_train['Source'].unique()))
        src_code = get_code(source)
        st.success(f"🛫 Source Code: **{src_code}**") 
        
        # Destination (Updates instantly now)
        destination = st.selectbox("Destination", sorted(x_train['Destination'].unique()))
        dest_code = get_code(destination)
        st.error(f"🛬 Destination Code: **{dest_code}**") 
        
        airline = st.selectbox("Airline", sorted(x_train['Airline'].unique()))
        
    with c2:
        st.subheader("Date & Time")
        if 'Month' in x_train.columns:
            months = sorted(x_train['Month'].unique())
        else:
            months = ['March', 'April', 'May', 'June', 'September', 'December']
        month = st.selectbox("Month", months)
        day_number = st.number_input("Day Number", 1, 31, 1)
        dept_hour = st.number_input("Departure Hour", 0, 23, 10)

    st.markdown("---")
    st.subheader("Route Selection")
    
    selected_route = None
    stops_val = 0
    
    if 'Route' in x_train.columns:
        # 1. Get all routes
        all_routes = sorted(x_train['Route'].unique().astype(str))
        
        # 2. FILTER: Show only routes starting with the exact Source Code
        # Now this runs immediately when you change 'Source' above
        filtered_routes = []
        for r in all_routes:
            parts = r.upper().replace("→", " ").replace("->", " ").split()
            if parts and parts[0] == src_code:
                filtered_routes.append(r)
        
        if filtered_routes:
            selected_route_raw = st.selectbox("Select Route", filtered_routes)
            
            # 3. VALIDATE
            if is_route_valid(selected_route_raw, src_code, dest_code):
                selected_route = selected_route_raw
                stops_val = route_lookup.get(selected_route, 0)
                st.metric("Total Stops", stops_val)
                st.success("✅ Valid Route")
            else:
                st.warning(f"⚠️ **{selected_route_raw}** starts at **{src_code}** but does not reach **{dest_code}**. Please check your destination.")
                selected_route = None
        else:
            st.error(f"No routes found starting with code **{src_code}**.")
    else:
        st.error("Route column missing.")

    # We only use a button for the final prediction calculation
    if st.button("Predict Price", type="primary"):
        if selected_route:
            # Prepare Input
            day_name = get_day_name(day_number, month)
            quarter = get_day_quarter(dept_hour)
            
            input_df = pd.DataFrame({
                'Airline': [airline], 'Source': [source], 'Destination': [destination],
                'Month': [month], 'Route': [selected_route],
                'Day_number': [day_number], 'Dept_hour': [dept_hour],
                'Day': [day_name], 'Dept_Day_Quarter': [quarter],
                'Total_Stops': [stops_val]
            })
            
            # Align Cols
            final_input = pd.DataFrame(columns=x_train.columns)
            for col in x_train.columns:
                final_input.loc[0, col] = input_df.iloc[0].get(col, 0)
                
            # Types
            for col in final_input.columns:
                if x_train[col].dtype == 'object':
                    final_input[col] = final_input[col].astype(str)
                else:
                    final_input[col] = pd.to_numeric(final_input[col])

            try:
                pred = model.predict(final_input)[0]
                st.success(f"### Estimated Price: ₹ {np.expm1(pred):,.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Please select a valid route.")

elif page == "📊 Model Evaluation":
    st.title("Model Evaluation")
    if st.button("Evaluate"):
        with st.spinner("Running..."):
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test_series, y_pred)
            st.metric("R2 Score", f"{r2:.4f}")
            
            fig, ax = plt.subplots()
            sns.scatterplot(x=np.expm1(y_test_series), y=np.expm1(y_pred), ax=ax)
            ax.plot([0, 80000], [0, 80000], 'r--')
            st.pyplot(fig)
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

# --- 2. Airport Codes (For Text Search) ---
CITY_TO_CODE = {
    'Banglore': 'BLR', 'Bangalore': 'BLR',
    'Delhi': 'DEL', 'New Delhi': 'DEL',
    'Kolkata': 'CCU', 'Calcutta': 'CCU',
    'Hyderabad': 'HYD', 'Chennai': 'MAA', 'Madras': 'MAA',
    'Mumbai': 'BOM', 'Bombay': 'BOM',
    'Cochin': 'COK', 'Kochi': 'COK',
    'Pune': 'PNQ', 'Goa': 'GOI', 'Jaipur': 'JAI',
    'Lucknow': 'LKO', 'Patna': 'PAT', 'Varanasi': 'VNS'
}

def get_code(city):
    return CITY_TO_CODE.get(city.strip(), city[:3].upper())

# Load Data
x_train, x_test, y_test = load_data()
model = load_model()

if x_train is None or model is None:
    st.stop()

if isinstance(y_test, pd.DataFrame):
    y_test_series = y_test.iloc[:, 0]
else:
    y_test_series = y_test

# Build Lookup for Stops
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
    
    # ---------------------------------------------------------
    # INPUTS
    # ---------------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Flight Info")
        
        # Source
        source = st.selectbox("Source", sorted(x_train['Source'].unique()))
        
        # Destination
        destination = st.selectbox("Destination", sorted(x_train['Destination'].unique()))
        
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
        # ---------------------------------------------------------
        # NEW FILTERING LOGIC
        # ---------------------------------------------------------
        # 1. Get ALL routes that start from this Source (using Source column)
        all_source_routes = x_train[x_train['Source'] == source]['Route'].unique()
        all_source_routes = sorted(all_source_routes.astype(str))
        
        # 2. Filter: Does the route string contain the Destination (Code or Name)?
        dest_code = get_code(destination)
        
        valid_routes = []
        for r in all_source_routes:
            # Check if 'BOM' or 'MUMBAI' is in 'DEL -> BOM -> COK'
            # We use .upper() for case insensitive check
            if (dest_code in r.upper()) or (destination.upper() in r.upper()):
                valid_routes.append(r)
        
        # 3. Display
        if len(valid_routes) > 0:
            selected_route = st.selectbox("Select Route", valid_routes)
            
            # Lookup hidden features
            stops_val = route_lookup.get(selected_route, 0)
            
            # Display Info
            c_a, c_b = st.columns(2)
            c_a.metric("Total Stops", stops_val)
            c_b.success("✅ Valid Route")
            
        else:
            st.error(f"No routes found starting at **{source}** that contain **{destination}** ({dest_code}).")
            selected_route = None
    else:
        st.error("Route column missing.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------
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
            
            # Align Columns
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
                real_price = np.expm1(pred)
                
                st.markdown(f"""
                <div style="background-color:#d4edda;padding:10px;border-radius:10px;border:2px solid #c3e6cb">
                    <h2 style="color:#155724;text-align:center;">🎫 Estimated Price: ₹ {real_price:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Cannot predict without a valid route.")

elif page == "📊 Model Evaluation":
    st.title("Model Evaluation")
    if st.button("Evaluate"):
        with st.spinner("Running..."):
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test_series, y_pred)
            st.metric("R2 Score", f"{r2:.4f}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=np.expm1(y_test_series), y=np.expm1(y_pred), ax=ax, alpha=0.6)
            min_val = min(np.expm1(y_test_series).min(), np.expm1(y_pred).min())
            max_val = max(np.expm1(y_test_series).max(), np.expm1(y_pred).max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            st.pyplot(fig)

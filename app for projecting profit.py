import streamlit as st
import os

project_directory = r"C:\Users\Asus\Documents\ValuTrades\Profit projection"
os.chdir(project_directory)
from profit_proj_try_2 import load_and_prepare_data, run_tactical_forecast

st.title("Tactical Profit Forecasting Tool")

@st.cache_data
def load_data():
    df, models, model_residuals = load_and_prepare_data()
    return df, models, model_residuals

df, models, model_residuals = load_data()

# Create user inputs in the sidebar
instrument = st.sidebar.selectbox("Select Instrument", df['Instrument'].unique())
region_choice = st.sidebar.selectbox("Select Region to Forecast", ['Total', 'China', 'RoW'])
regime = st.sidebar.selectbox("Select Starting Regime", df['Regime'].unique())
days = st.sidebar.slider("Days to Forecast", 1, 30, 5)
num_sims = st.sidebar.slider("Number of Simulations", 10, 5000, 1000)


# Create a button to run the forecast
if st.sidebar.button("Run Forecast"):
    with st.spinner("Running Monte Carlo simulation..."):
        # This function needs to be modified to return the figure object
        fig, summary_text = run_tactical_forecast(
        instrument=instrument,
        starting_regime=regime,
        forecast_days=days,
        forecast_region=region_choice,
        num_sims=num_sims,
        df=df,
        models=models,
        model_residuals=model_residuals
    )
    
    # Check if the forecast was successful before displaying results
    if fig is not None:
        st.header("Forecast Results")
        st.text(summary_text) # Display the text summary
        st.pyplot(fig)       # Display the plot directly on the web page
    else:
        st.error("Could not generate a forecast for the selected scenario. There might not be enough historical data for this specific instrument and regime.")
     
        
        st.header("Forecast Results")
        st.text(summary_text) # Display the text summary
        st.pyplot(fig)       # Display the plot directly on the web page
        
        # python -m streamlit run "app for projecting profit.py"
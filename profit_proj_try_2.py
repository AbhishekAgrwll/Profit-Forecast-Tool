import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
import os 
import seaborn as sns

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
DATA_FILES = [
    'price_profit_DE40Cash.csv',
    'price_profit_EURUSD.csv',
    'price_profit_USDJPY.csv',
    'price_profit_USOilCash.csv',
    'price_profit_XAUUSD.csv'
]

INSTRUMENTS = [os.path.basename(f).replace('price_profit_', '').replace('.csv', '') for f in DATA_FILES]
REGIONS = ['China', 'RoW']
N_SIMULATIONS = 1000
N_DAYS = 252

# =============================================================================
# STEP 1: DATA PREPARATION & FEATURE ENGINEERING
# =============================================================================
def load_and_prepare_data():
    print("--- Step 1: Loading Data and Engineering Features ---")

    all_data = []

    try:
        for file_path in DATA_FILES:
            instrument_name = os.path.basename(file_path).replace('price_profit_', '').replace('.csv', '')
            
            temp_df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True, thousands=',')
            
            temp_df['Instrument'] = instrument_name
            
            temp_df.columns = temp_df.columns.str.replace(f'{instrument_name}_', '')
            temp_df = temp_df.rename(columns={'profit': 'Total_Profit', 'lot': 'Total_Lot', 'World_profit': 'PnL_RoW', 'World_lot': 'Lots_RoW', 'China_profit': 'PnL_China', 'China_lot': 'Lots_China'})
            
            all_data.append(temp_df)

        df = pd.concat(all_data, ignore_index=True)
        
        df['Date'] = pd.to_datetime(df['Date'])


    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data files are in the correct directory.")
        df = pd.DataFrame() 

    if not df.empty:
        df['Daily_Return'] = (df['Price'] - df['Open']) / df['Open']
        df['Intraday_Volatility'] = (df['High'] - df['Low']) / df['Open']
        

        key_model_columns = ['PnL_China', 'Lots_China', 'PnL_RoW', 'Lots_RoW', 'Daily_Return', 'Intraday_Volatility']
        df.dropna(subset=key_model_columns, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True) 
        df.dropna(subset=key_model_columns, inplace=True)
        print("\n--- Data Cleaning Summary ---")
        print("Number of valid rows remaining per instrument:")
        print(df['Instrument'].value_counts())
        print("---------------------------\n")


        print("All data files loaded and merged successfully.")
        print("Master dataframe head:")
        print(df.head())
    else:
        print("Could not load data. Exiting.")
        exit()



    # =============================================================================
    # STEP 2: DEFINE MARKET REGIMES
    # =============================================================================
    print("\n--- Step 2: Defining Market Regimes ---")

    df['vol_regime'] = df.groupby('Instrument')['Intraday_Volatility'].transform(
        lambda x: pd.qcut(x, 2, labels=['Low_Vol', 'High_Vol'], duplicates='drop')
    )
    df['ret_regime'] = df.groupby('Instrument')['Daily_Return'].transform(
        lambda x: np.where(x >= 0, 'Up_Trend', 'Down_Trend')
    )


    df['Regime'] = df['vol_regime'].astype(str) + '_' + df['ret_regime'].astype(str)

    print("Market regimes defined. Example counts per instrument:")
    print(df.groupby('Instrument')['Regime'].value_counts().unstack().fillna(0))


    print("\n--- Visualizing Core Variable Correlations ---")


    correlation_vars = ['PnL_China', 'Lots_China', 'Daily_Return', 'Intraday_Volatility']


    for instrument in INSTRUMENTS:
        print(f"Generating correlation plot for {instrument}...")
        
        instrument_subset = df[df['Instrument'] == instrument]
        
        sns.pairplot(instrument_subset[correlation_vars].dropna())
        plt.suptitle(f'Correlation Matrix for {instrument}', y=1.02)
        plt.show()
        
    # =============================================================================
    # STEP 3: BUILD REGIME-BASED PREDICTIVE MODELS
    # =============================================================================
    print("\n--- Step 3: Building Regime-Based Regression Models ---")

    models = {}
    model_residuals = {}


    for instrument in INSTRUMENTS:
        models[instrument] = {}
        model_residuals[instrument] = {}
        for region in REGIONS:
            models[instrument][region] = {}
            model_residuals[instrument][region] = {}
            for regime in df['Regime'].unique():
                
                subset = df[(df['Instrument'] == instrument) & (df['Regime'] == regime)]
                
                if len(subset) < 10: 
                    continue
                pnl_col = f'PnL_{region}'
                lots_col = f'Lots_{region}'
                
                required_cols = [pnl_col, lots_col, 'Daily_Return', 'Intraday_Volatility']
                if not all(col in subset.columns for col in required_cols):
                    continue
                
                X = subset[[lots_col, 'Daily_Return', 'Intraday_Volatility']].dropna()
                y = subset.loc[X.index, pnl_col] 

                if len(X) < 10:
                    continue


                model = LinearRegression()
                model.fit(X, y)
                

                models[instrument][region][regime] = model
                predictions = model.predict(X)
                model_residuals[instrument][region][regime] = y - predictions

    print("Models built successfully for each instrument, region, and regime.")
    return df, models, model_residuals


# =============================================================================
# STEP 4: SCENARIO-BASED MONTE CARLO SIMULATION
# =============================================================================
def run_main_simulation(df, models, model_residuals):
    print("\n--- Step 4: Running Scenario-Based Monte Carlo Simulation ---")

    final_cumulative_pnl = np.zeros((N_DAYS + 1, N_SIMULATIONS))

    for i in range(N_SIMULATIONS):
        if (i + 1) % 100 == 0:
            print(f"Running simulation {i+1}/{N_SIMULATIONS}...")
            
        daily_total_pnl = []

        for day in range(N_DAYS):
            day_pnl = 0
            

            for instrument in INSTRUMENTS:
                sample_day = df[df['Instrument'] == instrument].sample(1).iloc[0]
                regime = sample_day['Regime']
                
                for region in REGIONS:

                    if regime in models.get(instrument, {}).get(region, {}):
                        
                        regime_subset = df[(df['Instrument'] == instrument) & (df['Regime'] == regime)]
                        if regime_subset.empty: continue
                        
                        simulated_drivers_sample = regime_subset.sample(1).iloc[0]
                        
                        lots_col = f'Lots_{region}'
                        sim_lots = simulated_drivers_sample[lots_col]
                        sim_ret = simulated_drivers_sample['Daily_Return']
                        sim_vol = simulated_drivers_sample['Intraday_Volatility']

                        X_pred = pd.DataFrame({
                            lots_col: [sim_lots],
                            'Daily_Return': [sim_ret],
                            'Intraday_Volatility': [sim_vol]
                        })
                        
                        model = models[instrument][region][regime]
                        predicted_pnl = model.predict(X_pred)[0]
                        
                        residuals = model_residuals[instrument][region][regime]
                        random_shock = np.random.choice(residuals) if not residuals.empty else 0
                        
                        simulated_pnl = predicted_pnl + random_shock
                        day_pnl += simulated_pnl

            daily_total_pnl.append(day_pnl)

        
        final_cumulative_pnl[1:, i] = np.cumsum(daily_total_pnl)

    print("Simulation complete.")

    # =============================================================================
    # FINAL RESULTS & VISUALIZATION
    # =============================================================================
    print("\n--- Final Results ---")


    final_day_profits = final_cumulative_pnl[-1, :]


    median_profit = np.median(final_day_profits)
    mean_profit = np.mean(final_day_profits)
    q1_profit = np.percentile(final_day_profits, 25)
    q3_profit = np.percentile(final_day_profits, 75)
    min_profit = np.min(final_day_profits)
    max_profit = np.max(final_day_profits)

    print(f"Median (Most Likely) 1-Year Cumulative Profit: ${median_profit:,.2f}")
    print(f"Mean (Average) 1-Year Cumulative Profit:       ${mean_profit:,.2f}")
    print(f"50% Confidence Interval (Q1-Q3):              ${q1_profit:,.2f} to ${q3_profit:,.2f}")
    print(f"Full Range (Min-Max):                         ${min_profit:,.2f} to ${max_profit:,.2f}")


    plt.figure(figsize=(12, 7))
    plt.plot(final_cumulative_pnl, color='purple', alpha=0.1)
    plt.title('Scenario-Based Monte Carlo Simulation: Total Cumulative Profit (1 Year)')
    plt.xlabel('Trading Days from Today')
    plt.ylabel('Simulated Cumulative Profit ($)')
    plt.grid(True)
    plt.savefig('monte_carlo_paths.png')
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.hist(final_day_profits, bins=50, color='navy', alpha=0.7)
    plt.axvline(median_profit, color='red', linestyle='--', label=f'Median: ${median_profit:,.0f}')
    plt.axvline(mean_profit, color='orange', linestyle='--', label=f'Mean: ${mean_profit:,.0f}')
    plt.title('Distribution of Final Cumulative Profit after 1 Year')
    plt.xlabel('Total Cumulative Profit ($)')
    plt.ylabel('Frequency (Number of Simulations)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig('final_profit_distribution.png')
    plt.close()

# =============================================================================
# STEP 5: TACTICAL FORECASTING FUNCTION 
# =============================================================================

def run_tactical_forecast(instrument, starting_regime, forecast_days, num_sims, df, models, model_residuals, forecast_region='Total'):
    """
    Runs a short-term, conditional Monte Carlo simulation for a specific instrument
    and starting market regime.
    """
    print(f"\n--- Running Tactical Forecast ---")
    print(f"Instrument: {instrument}, Starting Regime: {starting_regime}, Days: {forecast_days}\n")
    
    
    tactical_pnl = np.zeros((forecast_days + 1, num_sims))

    for i in range(num_sims):
        daily_instrument_pnl = []
        
        if forecast_region == 'China':
            regions_to_simulate = ['China']
        elif forecast_region == 'RoW':
            regions_to_simulate = ['RoW']
        else: # Default to 'Total'
            regions_to_simulate = REGIONS # REGIONS is ['China', 'RoW']
        
        for day in range(forecast_days):
            day_pnl = 0
            current_regime = starting_regime
            
            if day > 0:
                instrument_subset = df[df['Instrument'] == instrument]
                if not instrument_subset.empty:
                    current_regime = instrument_subset.sample(1).iloc[0]['Regime']

            for region in regions_to_simulate:
                if current_regime in models.get(instrument, {}).get(region, {}):
                    regime_subset = df[(df['Instrument'] == instrument) & (df['Regime'] == current_regime)]
                    if regime_subset.empty: continue
                    
                    sim_drivers = regime_subset.sample(1).iloc[0]
                    lots_col = f'Lots_{region}'
                    
                    X_pred = pd.DataFrame({
                        lots_col: [sim_drivers[lots_col]],
                        'Daily_Return': [sim_drivers['Daily_Return']],
                        'Intraday_Volatility': [sim_drivers['Intraday_Volatility']]
                    })
                    
                    model = models[instrument][region][current_regime]
                    predicted_pnl = model.predict(X_pred)[0]
                    residuals = model_residuals[instrument][region][current_regime]
                    random_shock = np.random.choice(residuals) if not residuals.empty else 0
                    day_pnl += predicted_pnl + random_shock
            
            daily_instrument_pnl.append(day_pnl)
            
        if daily_instrument_pnl:
            tactical_pnl[1:, i] = np.cumsum(daily_instrument_pnl)
            

    final_day_profits = tactical_pnl[-1, :]
    
    median_profit = np.median(final_day_profits)
    q1_profit = np.percentile(final_day_profits, 25)
    q3_profit = np.percentile(final_day_profits, 75)

    print(f"--- Tactical Forecast Results for {instrument} ({forecast_days} days) ---")
    print(f"Median Cumulative Profit: ${median_profit:,.2f}")
    print(f"50% Confidence Range: ${q1_profit:,.2f} to ${q3_profit:,.2f}\n")
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_day_profits, bins=30, color='teal', alpha=0.7)
    plt.title(f'Distribution of {forecast_days}-Day Profit for {instrument}\n(Starting in {starting_regime} Regime)')
    plt.xlabel(f'Cumulative Profit ($)')
    plt.ylabel('Frequency')
    plt.axvline(median_profit, color='red', linestyle='--', label=f'Median: ${median_profit:,.0f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_day_profits, bins=30, color='teal', alpha=0.7)
    ax.set_title(f'Distribution of {forecast_days}-Day Profit for {instrument}\n(Starting in {starting_regime} Regime)')
    ax.set_xlabel(f'Cumulative Profit ($)')
    ax.set_ylabel('Frequency')
    ax.axvline(median_profit, color='red', linestyle='--', label=f'Median: ${median_profit:,.0f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    summary_text = (
        f"--- Tactical Forecast Results for {instrument} ({forecast_days} days) ---\n"
        f"Median Cumulative Profit: ${median_profit:,.2f}\n"
        f"50% Confidence Range: ${q1_profit:,.2f} to ${q3_profit:,.2f}\n"
    )
    
    return fig, summary_text

# =============================================================================
# STEP 6: REGIME-CONDITIONAL VAR & CVAR FUNCTION (NEW FEATURE)
# =============================================================================

def calculate_regime_var_cvar(df, instrument, region, regime, var_period_days, confidence_level):
    """
    Calculates historical VaR and CVaR for a specific instrument, region,
    and market regime.
    """
    print(f"\n--- Calculating {var_period_days}-Day VaR & CVaR for {regime} ---")
    
    # Determine the PnL column to use
    if region == 'Total':
        pnl_col = 'Total_Profit'
    else:
        pnl_col = f'PnL_{region}'
        
    # Filter for the specific instrument AND market regime
    subset_df = df[(df['Instrument'] == instrument) & (df['Regime'] == regime)].copy()
    
    if len(subset_df) < var_period_days:
        error_msg = f"Not enough data for {instrument} in the {regime} regime to calculate a {var_period_days}-day VaR."
        return error_msg, None

    # Calculate the rolling sum of PnL over the specified period
    subset_df['rolling_pnl'] = subset_df[pnl_col].rolling(window=var_period_days).sum()
    historical_outcomes = subset_df['rolling_pnl'].dropna()
    
    if historical_outcomes.empty:
        return f"Not enough rolling periods for the calculation.", None
        
    # --- Calculate VaR (the threshold) ---
    var_percentile = 100 - confidence_level
    var_value = np.percentile(historical_outcomes, var_percentile)
    
    # --- Calculate CVaR (the expected shortfall) ---
    # Find all losses that were worse than the VaR threshold
    tail_losses = historical_outcomes[historical_outcomes <= var_value]
    cvar_value = tail_losses.mean() if not tail_losses.empty else var_value

    # --- Create Summary Text ---
    summary_text = (
        f"Regime-Conditional Risk for {instrument} ({region}) in '{regime}' state:\n\n"
        f"Value at Risk (VaR) at {confidence_level}% confidence:\n"
        f"  - Over any {var_period_days}-day period in this regime, there is a {100-confidence_level}% chance of losing more than ${-var_value:,.2f}.\n\n"
        f"Conditional VaR (Expected Shortfall) at {confidence_level}% confidence:\n"
        f"  - In the scenarios where you DO lose more than the VaR amount, the AVERAGE loss is expected to be ${-cvar_value:,.2f}."
    )
    
    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(historical_outcomes, bins=50, color='firebrick', alpha=0.7, label='Historical Outcomes')
    ax.axvline(var_value, color='black', linestyle='--', lw=2, label=f'VaR Threshold: ${var_value:,.0f}')
    ax.axvline(cvar_value, color='yellow', linestyle='-', lw=2, label=f'CVaR (Expected Shortfall): ${cvar_value:,.0f}')
    ax.set_title(f'Historical {var_period_days}-Day P&L in "{regime}" Regime\n({instrument} - {region})')
    ax.set_xlabel(f'Cumulative {var_period_days}-Day Profit/Loss ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    
    return summary_text, fig

# =============================================================================
# SCRIPT EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    
    # This code will now ONLY run when you execute this file directly.
    
    # Set the working directory first
    project_directory = r"C:\Users\Asus\Documents\ValuTrades\Profit projection"
    os.chdir(project_directory)

    # 1. Run the main data preparation and model building process
    df, models, model_residuals = load_and_prepare_data()
    
    # 2. Run the main annual simulation
    run_main_simulation(df, models, model_residuals)

    # 3. Now you can run your tactical forecasts for testing
    print("\n--- Running Example Tactical Forecasts ---")
    
    if not df.empty:
        # Scenario 1
        run_tactical_forecast(
            instrument='XAUUSD', 
            starting_regime='High_Vol_Down_Trend', 
            forecast_days=5, 
            num_sims=5000,
            df=df,
            models=models,
            model_residuals=model_residuals
        )

        # Scenario 2
        run_tactical_forecast(
            instrument='DE40Cash', 
            starting_regime='Low_Vol_Up_Trend', 
            forecast_days=10, 
            num_sims=5000,
            df=df,
            models=models,
            model_residuals=model_residuals
        )



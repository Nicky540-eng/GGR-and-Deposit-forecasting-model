import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

def load_and_synchronize_datasets():
    print("[1/5] Extracting CSV files and filtering from 2025 onwards...")
    ggr_df = pd.read_csv('GGR_Cleaned.csv', parse_dates=['Year-Month'])
    dep_df = pd.read_csv('Deposits_Cleaned.csv', parse_dates=['Year-Month'])
    ftd_df = pd.read_csv('FTDS_Cleaned.csv', parse_dates=['Year-Month'])
    reg_df = pd.read_csv('Registrations_Cleaned.csv', parse_dates=['Year-Month'])
    
    ggr_df = ggr_df[['Year-Month', 'GGR']]
    dep_df = dep_df[['Year-Month', 'Deposits']]
    ftd_df = ftd_df[['Year-Month', 'FTDs']]
    reg_df = reg_df[['Year-Month', 'Registration']].rename(columns={'Registration': 'Registrations'})
    
    df = ggr_df.merge(dep_df, on='Year-Month', how='inner')
    df = df.merge(ftd_df, on='Year-Month', how='inner')
    df = df.merge(reg_df, on='Year-Month', how='inner')
    
    df.set_index('Year-Month', inplace=True)
    df = df.sort_index()
    
    # Exclude 2024 to prevent underfitting models
    df = df.loc[df.index >= '2025-01-01']
    
    for col in ['GGR', 'Deposits', 'Registrations', 'FTDs']:
        df[col] = df[col].astype(float)
        
    return df

def generate_engineered_features(df):
    df = df.copy()
    df['FTDs_Lag_1'] = df['FTDs'].shift(1)
    df['FTDs_Lag_2'] = df['FTDs'].shift(2)
    df['Reg_to_FTD_Ratio'] = df['FTDs'] / (df['Registrations'] + 1e-5)
    df['GGR_3M_Rolling'] = df['GGR'].shift(1).rolling(window=3).mean()
    df['Deposits_3M_Rolling'] = df['Deposits'].shift(1).rolling(window=3).mean()
    return df

def run_main_forecasting_pipeline():
    base_df = load_and_synchronize_datasets()
    featured_df = generate_engineered_features(base_df).dropna()
    
    # Strictly lock down accuracy backtesting to Q1 2026 actual entries
    train_2025 = featured_df.loc[featured_df.index < '2026-01-01']
    q1_2026_actuals = featured_df.loc[(featured_df.index >= '2026-01-01') & (featured_df.index <= '2026-03-01')]
    
    features_list = ['FTDs_Lag_1', 'FTDs_Lag_2', 'Reg_to_FTD_Ratio', 'GGR_3M_Rolling', 'Deposits_3M_Rolling']
    results_registry = {}
    
    print("[2/5] Running backtests strictly locked to Q1 2026 actuals...")
    for target in ['GGR', 'Deposits']:
        lgb_aud = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        lgb_aud.fit(train_2025[features_list], train_2025[target])
        lgb_err = mean_absolute_percentage_error(q1_2026_actuals[target], lgb_aud.predict(q1_2026_actuals[features_list]))
        
        sari_aud = SARIMAX(train_2025[target], exog=train_2025[['Registrations', 'FTDs']], order=(1,0,0))
        sari_aud_fit = sari_aud.fit(disp=False)
        sari_preds = sari_aud_fit.forecast(steps=len(q1_2026_actuals), exog=q1_2026_actuals[['Registrations', 'FTDs']])
        sari_err = mean_absolute_percentage_error(q1_2026_actuals[target], sari_preds)
        
        # Explicitly label which MAPE score belongs to which model architecture
        results_registry[target] = {
            'LightGBM_MAPE': float(lgb_err),
            'SARIMAX_MAPE': float(sari_err)
        }

    print("[3/5] Constructing projection timeline canvas...")
    future_horizon = pd.date_range(start='2026-01-01', end='2027-12-01', freq='MS')
    max_historical_date = base_df.index.max()  # April 2026
    
    lgb_canvas = base_df.copy()
    sari_canvas = base_df.copy()
    
    target_mom_growth = 0.015  # 1.5% MoM trend growth compounder
    running_reg = base_df.loc[max_historical_date, 'Registrations']
    running_ftd = base_df.loc[max_historical_date, 'FTDs']
    
    for month in future_horizon:
        if month not in lgb_canvas.index:
            running_reg *= (1 + target_mom_growth)
            running_ftd *= (1 + target_mom_growth)
            lgb_canvas.loc[month, ['Registrations', 'FTDs', 'GGR', 'Deposits']] = [running_reg, running_ftd, np.nan, np.nan]
            sari_canvas.loc[month, ['Registrations', 'FTDs', 'GGR', 'Deposits']] = [running_reg, running_ftd, np.nan, np.nan]

    print("[4/5] Running independent recursive forecasting models...")
    for target in ['GGR', 'Deposits']:
        # Fit models on the complete historical track
        prod_feat = generate_engineered_features(base_df).dropna()
        final_lgb = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        final_lgb.fit(prod_feat[features_list], prod_feat[target])
        
        final_sari = SARIMAX(base_df[target], exog=base_df[['Registrations', 'FTDs']], order=(1,0,0))
        final_sari_fit = final_sari.fit(disp=False)
        
        # Populate early 2026 with independent predictions to show model variations
        for month in future_horizon:
            if month <= max_historical_date:
                # Store independent, un-trended pure model evaluations for audit months
                lgb_feat = generate_engineered_features(lgb_canvas)
                lgb_canvas.loc[month, target] = final_lgb.predict(lgb_feat.loc[[month]][features_list])[0]
                
                steps_back = list(future_horizon).index(month) + 1
                sari_canvas.loc[month, target] = final_sari_fit.forecast(steps=steps_back, exog=base_df.loc[base_df.index.isin(future_horizon[:steps_back]), ['Registrations', 'FTDs']]).values[-1]
            else:
                # Out-of-sample forward horizon gets the true compounding velocity expansion
                future_months_list = [m for m in future_horizon if m > max_historical_date]
                steps_forward = future_months_list.index(month) + 1
                compound_multiplier = (1 + target_mom_growth) ** steps_forward
                
                # Baseline anchors to protect models against long-range decay
                base_anchor_lgb = lgb_canvas.loc[max_historical_date, target]
                base_anchor_sari = sari_canvas.loc[max_historical_date, target]
                
                lgb_canvas.loc[month, target] = base_anchor_lgb * compound_multiplier
                sari_canvas.loc[month, target] = base_anchor_sari * compound_multiplier

    print("[5/5] Packaging comparison paths into database payload...")
    payload = {
        'lgb_projections': lgb_canvas.loc[lgb_canvas.index >= '2026-01-01'][['GGR', 'Deposits']],
        'sari_projections': sari_canvas.loc[sari_canvas.index >= '2026-01-01'][['GGR', 'Deposits']],
        'actuals_2026': base_df.loc[base_df.index >= '2026-01-01'][['GGR', 'Deposits']],
        'metrics': results_registry
    }
    
    with open('live_growth_payload.pkl', 'wb') as f:
        pickle.dump(payload, f)
        
    print("✨ Model Engine Completed Successfully.")

if __name__ == "__main__":
    run_main_forecasting_pipeline()
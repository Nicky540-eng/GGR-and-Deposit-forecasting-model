import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="GGR & Deposit Forecasting Model", layout="wide")

def load_payload():
    if not os.path.exists('live_growth_payload.pkl'):
        return None
    with open('live_growth_payload.pkl', 'rb') as f:
        return pickle.load(f)

payload = load_payload()
if payload is None:
    st.error("❌ Data Payload asset missing. Run forecast_engine.py first.")
    st.stop()

lgb_df = payload['lgb_projections']
sari_df = payload['sari_projections']
actuals = payload['actuals_2026']
metrics = payload['metrics']

st.title("📊 GGR & Deposit Forecasting Model")
st.markdown("### Benchmarked Strictly Against Q1 2026 Operational Milestones")
st.markdown("---")

selected_kpi = st.sidebar.selectbox("Select Target Metric:", ["GGR", "Deposits"])
selected_horizon = st.sidebar.radio("View Range Selection:", ["Full Horizon View (2026 - 2027)", "Year 2026 Matrix Only", "Year 2027 Projections Only"])

# -------------------------------------------------------------------------
# DYNAMIC KPI CALCULATOR FOR THE TOP SUMMARY CARDS
# -------------------------------------------------------------------------
q1_dates = pd.date_range(start='2026-01-01', end='2026-03-01', freq='MS')

q1_actual_sum = actuals.loc[actuals.index.isin(q1_dates), selected_kpi].sum()
q1_lgb_sum = lgb_df.loc[lgb_df.index.isin(q1_dates), selected_kpi].sum()
q1_sari_sum = sari_df.loc[sari_df.index.isin(q1_dates), selected_kpi].sum()

lgb_mape = metrics[selected_kpi]['LightGBM_MAPE']
sari_mape = metrics[selected_kpi]['SARIMAX_MAPE']

# Calculate variances for Q1 (complete data)
lgb_variance_q1 = q1_lgb_sum - q1_actual_sum
sari_variance_q1 = q1_sari_sum - q1_actual_sum
lgb_variance_millions_q1 = lgb_variance_q1 / 1_000_000
sari_variance_millions_q1 = sari_variance_q1 / 1_000_000
lgb_variance_pct_q1 = (lgb_variance_q1 / q1_actual_sum) * 100 if q1_actual_sum != 0 else 0
sari_variance_pct_q1 = (sari_variance_q1 / q1_actual_sum) * 100 if q1_actual_sum != 0 else 0

# -------------------------------------------------------------------------
# EXECUTIVE SUMMARY GRID LAYOUT - 4 COLUMNS SIDE-BY-SIDE
# -------------------------------------------------------------------------
st.header(f"🎯 Q1 2026 Performance Scorecard: {selected_kpi}")

card_style = """
    <div style="border: 2px solid #E2E8F0; padding: 18px; border-radius: 6px; 
                background-color: #F8FAFC; text-align: center; box-shadow: 1px 1px 4px rgba(0,0,0,0.02); height: 100px; display: flex; flex-direction: column; justify-content: center;">
        <p style="margin: 0; font-size: 13px; color: #64748B; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">{label}</p>
        <p style="margin: 6px 0 0 0; font-size: 22px; font-weight: bold; color: {color};">{value}</p>
    </div>
"""

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(card_style.format(label="Total Actuals Q1", value=f"R {q1_actual_sum:,.2f}", color="#0F172A"), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(label="Total LightGBM Forecast Q1", value=f"R {q1_lgb_sum:,.2f}", color="#2563EB"), unsafe_allow_html=True)

with col3:
    st.markdown(card_style.format(label="Total SARIMAX Forecast Q1", value=f"R {q1_sari_sum:,.2f}", color="#DC2626"), unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div style="border: 2px solid #E2E8F0; padding: 12px 18px; border-radius: 6px; 
                    background-color: #F8FAFC; text-align: center; box-shadow: 1px 1px 4px rgba(0,0,0,0.02); height: 100px;">
            <p style="margin: 0 0 4px 0; font-size: 12px; color: #64748B; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">Selected Production Engine MAPE</p>
            <p style="margin: 0; font-size: 14px; color: #2563EB; font-weight: normal;">
                LightGBM Error: <strong style="font-size: 16px;">{lgb_mape:.2%}</strong>
            </p>
            <p style="margin: 2px 0 0 0; font-size: 14px; color: #DC2626; font-weight: normal;">
                SARIMAX Error: <strong style="font-size: 16px;">{sari_mape:.2%}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Graph Filtering Ranges
if selected_horizon == "Year 2026 Matrix Only":
    plot_years = [2026]
elif selected_horizon == "Year 2027 Projections Only":
    plot_years = [2027]
else:
    plot_years = [2026, 2027]

# -------------------------------------------------------------------------
# CHART PLOTTING
# -------------------------------------------------------------------------
st.header(f"📈 Strategic Projections Pathway: {selected_kpi}")
fig = go.Figure()

filter_lgb = lgb_df.loc[lgb_df.index.year.isin(plot_years)]
filter_sari = sari_df.loc[sari_df.index.year.isin(plot_years)]
filter_act = actuals.loc[actuals.index.year.isin(plot_years)]

fig.add_trace(go.Scatter(x=filter_lgb.index, y=filter_lgb[selected_kpi], mode='lines+markers', name='LightGBM Prediction Pathway', line=dict(color='#2563EB', width=3)))
fig.add_trace(go.Scatter(x=filter_sari.index, y=filter_sari[selected_kpi], mode='lines+markers', name='SARIMAX Prediction Pathway', line=dict(color='#DC2626', width=3, dash='dash')))

display_act = filter_act.loc[filter_act[selected_kpi].notna()]
if len(display_act) > 0:
    fig.add_trace(go.Scatter(x=display_act.index, y=display_act[selected_kpi], mode='markers', name='Observed Historical Actuals', marker=dict(color='#0F172A', size=12, symbol='square')))

fig.update_layout(
    plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF', hovermode='x unified',
    xaxis=dict(showgrid=True, gridcolor='#E2E8F0', tickformat='%b %Y'),
    yaxis=dict(title="South African Rand (ZAR)", showgrid=True, gridcolor='#E2E8F0', tickprefix="R "),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# DATAFRAME - HONEST VARIANCE CALCULATIONS (ONLY WHERE DATA IS COMPLETE)
# -------------------------------------------------------------------------
st.header("💼 Corporate Performance Ledger Sheets")

def format_millions_with_comma(value):
    """Format value (already in millions) with comma as decimal separator"""
    if pd.isna(value):
        return None
    return f"{abs(value):.8f}".replace('.', ',')

def format_percentage_standard(value):
    """Format percentage in standard form (already multiplied by 100) with comma as decimal separator"""
    if pd.isna(value):
        return None
    return f"{value:.4f}".replace('.', ',')

def get_ledger_dataframe_with_variances(years_filter, kpi):
    """Generate dataframe with variance - ONLY where ALL months in period have actuals"""
    matrix_rows = []
    
    for yr in years_filter:
        year_total, year_lgb, year_sari = 0.0, 0.0, 0.0
        year_complete = True
        year_months_with_actuals = 0
        year_total_months = 12 if yr == 2026 else 12  # Assuming full years
        
        # Track which months have actuals for the year
        months_with_actuals = {}
        
        for qtr in [1, 2, 3, 4]:
            qtr_months = range((qtr-1)*3 + 1, qtr*3 + 1)
            qtr_act, qtr_lgb, qtr_sari = 0.0, 0.0, 0.0
            qtr_complete = True
            qtr_months_with_actuals = []
            
            months_data = []
            
            for mo in qtr_months:
                dt = pd.Timestamp(year=yr, month=mo, day=1)
                m_name = dt.strftime('%B %Y')
                
                v_act = actuals.loc[dt, kpi] if dt in actuals.index and not pd.isna(actuals.loc[dt, kpi]) else np.nan
                v_lgb = lgb_df.loc[dt, kpi] if dt in lgb_df.index else 0.0
                v_sari = sari_df.loc[dt, kpi] if dt in sari_df.index else 0.0
                
                has_actual = not pd.isna(v_act)
                
                # For individual months: Calculate variance if actual exists
                if has_actual:
                    lgb_var_millions = (v_lgb - v_act) / 1_000_000
                    lgb_var_pct = ((v_lgb - v_act) / v_act) * 100 if v_act != 0 else np.nan
                    sari_var_millions = (v_sari - v_act) / 1_000_000
                    sari_var_pct = ((v_sari - v_act) / v_act) * 100 if v_act != 0 else np.nan
                    months_with_actuals[mo] = True
                    year_months_with_actuals += 1
                else:
                    lgb_var_millions = np.nan
                    lgb_var_pct = np.nan
                    sari_var_millions = np.nan
                    sari_var_pct = np.nan
                    months_with_actuals[mo] = False
                    qtr_complete = False
                    year_complete = False
                
                months_data.append({
                    'period': m_name,
                    'actual': v_act,
                    'lgb': v_lgb,
                    'sari': v_sari,
                    'lgb_var_millions': lgb_var_millions,
                    'lgb_var_pct': lgb_var_pct,
                    'sari_var_millions': sari_var_millions,
                    'sari_var_pct': sari_var_pct,
                    'has_actual': has_actual
                })
                
                if has_actual:
                    qtr_act += v_act
                    qtr_months_with_actuals.append(mo)
                qtr_lgb += v_lgb
                qtr_sari += v_sari
            
            # Add monthly rows
            for month_data in months_data:
                matrix_rows.append({
                    'Period Timeline': month_data['period'],
                    'Observed Actuals': month_data['actual'],
                    'LightGBM Forecast': month_data['lgb'],
                    'SARIMAX Forecast': month_data['sari'],
                    'LightGBM Variance (R Millions)': month_data['lgb_var_millions'],
                    'LightGBM Variance %': month_data['lgb_var_pct'],
                    'SARIMAX Variance (R Millions)': month_data['sari_var_millions'],
                    'SARIMAX Variance %': month_data['sari_var_pct']
                })
            
            # Calculate quarter variances - ONLY if ALL 3 months have actuals
            if qtr_complete and len(qtr_months_with_actuals) == 3:
                qtr_lgb_var_millions = (qtr_lgb - qtr_act) / 1_000_000
                qtr_lgb_var_pct = ((qtr_lgb - qtr_act) / qtr_act) * 100 if qtr_act != 0 else np.nan
                qtr_sari_var_millions = (qtr_sari - qtr_act) / 1_000_000
                qtr_sari_var_pct = ((qtr_sari - qtr_act) / qtr_act) * 100 if qtr_act != 0 else np.nan
            else:
                qtr_lgb_var_millions = np.nan
                qtr_lgb_var_pct = np.nan
                qtr_sari_var_millions = np.nan
                qtr_sari_var_pct = np.nan
            
            matrix_rows.append({
                'Period Timeline': f'📊 QUARTER {qtr} TOTAL (Q{qtr})',
                'Observed Actuals': qtr_act if qtr_months_with_actuals else np.nan,
                'LightGBM Forecast': qtr_lgb,
                'SARIMAX Forecast': qtr_sari,
                'LightGBM Variance (R Millions)': qtr_lgb_var_millions,
                'LightGBM Variance %': qtr_lgb_var_pct,
                'SARIMAX Variance (R Millions)': qtr_sari_var_millions,
                'SARIMAX Variance %': qtr_sari_var_pct
            })
            
            if qtr_complete:
                year_total += qtr_act
            year_lgb += qtr_lgb
            year_sari += qtr_sari
        
        # Calculate year variances - ONLY if ALL 12 months have actuals
        if year_complete and year_months_with_actuals == 12:
            yr_lgb_var_millions = (year_lgb - year_total) / 1_000_000
            yr_lgb_var_pct = ((year_lgb - year_total) / year_total) * 100 if year_total != 0 else np.nan
            yr_sari_var_millions = (year_sari - year_total) / 1_000_000
            yr_sari_var_pct = ((year_sari - year_total) / year_total) * 100 if year_total != 0 else np.nan
        else:
            yr_lgb_var_millions = np.nan
            yr_lgb_var_pct = np.nan
            yr_sari_var_millions = np.nan
            yr_sari_var_pct = np.nan
        
        matrix_rows.append({
            'Period Timeline': f'🏆 YEAR {yr} GRAND TOTAL',
            'Observed Actuals': year_total if year_months_with_actuals > 0 else np.nan,
            'LightGBM Forecast': year_lgb,
            'SARIMAX Forecast': year_sari,
            'LightGBM Variance (R Millions)': yr_lgb_var_millions,
            'LightGBM Variance %': yr_lgb_var_pct,
            'SARIMAX Variance (R Millions)': yr_sari_var_millions,
            'SARIMAX Variance %': yr_sari_var_pct
        })
        
    return pd.DataFrame(matrix_rows)

# Generate the dataframe
df_ledger = get_ledger_dataframe_with_variances(plot_years, selected_kpi)

# Format the display dataframe
display_df = df_ledger.copy()
display_df['Observed Actuals'] = display_df['Observed Actuals'].apply(lambda x: f"R {x:,.2f}" if pd.notna(x) else "—")
display_df['LightGBM Forecast'] = display_df['LightGBM Forecast'].apply(lambda x: f"R {x:,.2f}" if pd.notna(x) else "—")
display_df['SARIMAX Forecast'] = display_df['SARIMAX Forecast'].apply(lambda x: f"R {x:,.2f}" if pd.notna(x) else "—")

# Create variance display strings for LightGBM
def format_lgb_variance(row):
    millions = row['LightGBM Variance (R Millions)']
    pct = row['LightGBM Variance %']
    
    if pd.isna(millions):
        return "—"
    else:
        arrow = "▲" if millions > 0 else "▼"
        millions_str = format_millions_with_comma(millions)
        pct_str = format_percentage_standard(abs(pct))
        return f"{arrow} {millions_str} ({pct_str})"

# Create variance display strings for SARIMAX
def format_sari_variance(row):
    millions = row['SARIMAX Variance (R Millions)']
    pct = row['SARIMAX Variance %']
    
    if pd.isna(millions):
        return "—"
    else:
        arrow = "▲" if millions > 0 else "▼"
        millions_str = format_millions_with_comma(millions)
        pct_str = format_percentage_standard(abs(pct))
        return f"{arrow} {millions_str} ({pct_str})"

display_df['LightGBM Variance'] = df_ledger.apply(format_lgb_variance, axis=1)
display_df['SARIMAX Variance'] = df_ledger.apply(format_sari_variance, axis=1)

# Select final columns
final_df = display_df[['Period Timeline', 'Observed Actuals', 'LightGBM Forecast', 'SARIMAX Forecast', 'LightGBM Variance', 'SARIMAX Variance']]

# Apply color styling function
def color_variance(val):
    if val == "—" or pd.isna(val):
        return ''
    if '▲' in str(val):
        return 'color: #10B981'  # Green for positive
    elif '▼' in str(val):
        return 'color: #EF4444'  # Red for negative
    return ''

# Apply styling to the dataframe
styled_df = final_df.style.map(color_variance, subset=['LightGBM Variance', 'SARIMAX Variance'])

# Display with st.dataframe
st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    height=800,
    column_config={
        "Period Timeline": st.column_config.TextColumn("Period Timeline", width="medium"),
        "Observed Actuals": st.column_config.TextColumn("Observed Actuals", width="small"),
        "LightGBM Forecast": st.column_config.TextColumn("LightGBM Forecast", width="small"),
        "SARIMAX Forecast": st.column_config.TextColumn("SARIMAX Forecast", width="small"),
        "LightGBM Variance": st.column_config.TextColumn("LightGBM Variance (R Millions / %)", width="medium"),
        "SARIMAX Variance": st.column_config.TextColumn("SARIMAX Variance (R Millions / %)", width="medium")
    }
)

# -------------------------------------------------------------------------
# ADDITIONAL EXECUTIVE SUMMARY WITH VARIANCE METRICS
# -------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎯 Variance Analysis Summary (Complete Data Only)")

col_v1, col_v2, col_v3 = st.columns(3)

with col_v1:
    st.metric(
        label="LightGBM Forecast Variance (Q1 2026)",
        value=f"{abs(lgb_variance_millions_q1):.8f}".replace('.', ','),
        delta=f"{lgb_variance_pct_q1:+.4f}".replace('.', ','),
        delta_color="normal"
    )

with col_v2:
    st.metric(
        label="SARIMAX Forecast Variance (Q1 2026)",
        value=f"{abs(sari_variance_millions_q1):.8f}".replace('.', ','),
        delta=f"{sari_variance_pct_q1:+.4f}".replace('.', ','),
        delta_color="normal"
    )

with col_v3:
    better_model = "LightGBM" if abs(lgb_variance_millions_q1) < abs(sari_variance_millions_q1) else "SARIMAX"
    better_variance = min(abs(lgb_variance_millions_q1), abs(sari_variance_millions_q1))
    st.metric(
        label="🏆 Best Performing Model (Q1 2026)",
        value=better_model,
        delta=f"{better_variance:.8f}".replace('.', ',')
    )

st.markdown("---")
st.caption("📌 **Note:** Variance calculations are shown ONLY where complete data exists (all months in the period have actuals). Individual months show variance when actual data exists. Quarter and Year totals only show variance when ALL months in that period have actual data.")
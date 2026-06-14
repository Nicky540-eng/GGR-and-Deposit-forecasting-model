import os

code_content = """import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Executive Projection Analytics", layout="wide")

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

st.title("📊 Executive Model Performance & Projections Hub")
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

# -------------------------------------------------------------------------
# EXECUTIVE SUMMARY GRID LAYOUT - 4 COLUMNS SIDE-BY-SIDE
# -------------------------------------------------------------------------
st.header(f"🎯 Q1 2026 Performance Scorecard: {selected_kpi}")

card_style = \"\"\"
    <div style="border: 2px solid #E2E8F0; padding: 18px; border-radius: 6px; 
                background-color: #F8FAFC; text-align: center; box-shadow: 1px 1px 4px rgba(0,0,0,0.02); height: 100px; display: flex; flex-direction: column; justify-content: center;">
        <p style="margin: 0; font-size: 13px; color: #64748B; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">{label}</p>
        <p style="margin: 6px 0 0 0; font-size: 22px; font-weight: bold; color: {color};">{value}</p>
    </div>
\"\"\"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(card_style.format(label="Total Actuals Q1", value=f"R {q1_actual_sum:,.2f}", color="#0F172A"), unsafe_allow_html=True)

with col2:
    st.markdown(card_style.format(label="Total LightGBM Forecast Q1", value=f"R {q1_lgb_sum:,.2f}", color="#2563EB"), unsafe_allow_html=True)

with col3:
    st.markdown(card_style.format(label="Total SARIMAX Forecast Q1", value=f"R {q1_sari_sum:,.2f}", color="#DC2626"), unsafe_allow_html=True)

with col4:
    st.markdown(f\"\"\"
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
    \"\"\", unsafe_allow_html=True)

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
# FINANCIAL LEDGER SHEETS GENERATOR (CRISP HTML CONVERSIONS)
# -------------------------------------------------------------------------
st.header("💼 Corporate Performance Ledger Sheets")

def construct_html_ledger_table(years_filter, kpi):
    ledger_rows = []
    
    for yr in years_filter:
        year_total, year_lgb_total, year_sari_total = 0.0, 0.0, 0.0
        
        for qtr in [1, 2, 3, 4]:
            qtr_months = range((qtr-1)*3 + 1, qtr*3 + 1)
            qtr_actual_sum, qtr_lgb_sum, qtr_sari_sum = 0.0, 0.0, 0.0
            has_qtr_actuals = False
            
            for mo in qtr_months:
                dt = pd.Timestamp(year=yr, month=mo, day=1)
                m_name = dt.strftime('%B %Y')
                
                val_act = actuals.loc[dt, kpi] if dt in actuals.index and not pd.isna(actuals.loc[dt, kpi]) else np.nan
                val_lgb = lgb_df.loc[dt, kpi] if dt in lgb_df.index else 0.0
                val_sari = sari_df.loc[dt, kpi] if dt in sari_df.index else 0.0
                
                str_act = f"R {val_act:,.2f}" if not pd.isna(val_act) else "—"
                str_lgb = f"R {val_lgb:,.2f}"
                str_sari = f"R {val_sari:,.2f}"
                
                ledger_rows.append([m_name, str_act, str_lgb, str_sari, "normal"])
                
                if not pd.isna(val_act):
                    qtr_actual_sum += val_act
                    has_qtr_actuals = True
                qtr_lgb_sum += val_lgb
                qtr_sari_sum += val_sari
            
            str_q_act = f"R {qtr_actual_sum:,.2f}" if has_qtr_actuals else "—"
            ledger_rows.append([f"TOTAL QUARTER {qtr} (Q{qtr})", str_q_act, f"R {qtr_lgb_sum:,.2f}", f"R {qtr_sari_sum:,.2f}", "subtotal"])
            
            year_total += qtr_actual_sum
            year_lgb_total += qtr_lgb_sum
            year_sari_total += qtr_sari_sum
            
        str_y_act = f"R {year_total:,.2f}" if year_total > 0 else "—"
        ledger_rows.append([f"🏆 GRAND TOTAL YEAR {yr}", str_y_act, f"R {year_lgb_total:,.2f}", f"R {year_sari_total:,.2f}", "grandtotal"])
        
    html_output = \"\"\"
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; color: #0F172A; margin-top: 15px;">
        <thead>
            <tr style="border-bottom: 2px solid #0F172A; background-color: #F8FAFC;">
                <th style="padding: 12px; font-weight: bold; width: 25%; text-align: left;">Period Timeline</th>
                <th style="padding: 12px; font-weight: bold; width: 25%; text-align: right;">Observed Actuals</th>
                <th style="padding: 12px; font-weight: bold; width: 25%; text-align: right;">LightGBM Forecast</th>
                <th style="padding: 12px; font-weight: bold; width: 25%; text-align: right;">SARIMAX Forecast</th>
            </tr>
        </thead>
        <tbody>
    \"\"\"
    
    for row in ledger_rows:
        timeline, act, lgb_val, sari_val, row_type = row
        
        if row_type == "subtotal":
            style_str = "background-color: #F8FAFC; font-weight: bold; border-top: 1px solid #CBD5E1; border-bottom: 1px solid #CBD5E1;"
        elif row_type == "grandtotal":
            style_str = "background-color: #F1F5F9; font-weight: bold; border-top: 2px solid #475569; border-bottom: 3px double #475569;"
        else:
            style_str = "border-bottom: 1px solid #E2E8F0;"
            
        html_output += f\"\"\"
        <tr style="{style_str}">
            <td style="padding: 10px 12px; text-align: left;">{timeline}</td>
            <td style="padding: 10px 12px; text-align: right;">{act}</td>
            <td style="padding: 10px 12px; text-align: right;">{lgb_val}</td>
            <td style="padding: 10px 12px; text-align: right;">{sari_val}</td>
        </tr>
        \"\"\"
        
    html_output += "</tbody></table>"
    return html_output

clean_html_table = construct_html_ledger_table(plot_years, selected_kpi)
st.markdown(clean_html_table, unsafe_allow_html=True)
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code_content)
print("🎯 Hard reset successful! 'app.py' has been forcefully overwritten directly via the system drive.")
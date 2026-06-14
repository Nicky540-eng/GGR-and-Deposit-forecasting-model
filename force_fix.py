import os

fixed_code = """import streamlit as st
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
# FINANCIAL LEDGER LOGIC WITH FIXED INLINE HTML COLORING
# -------------------------------------------------------------------------
def color_variance_value(actual_val, forecast_val, is_percentage=False):
    if pd.isna(actual_val) or actual_val == 0 or pd.isna(forecast_val) or forecast_val == 0:
        return "—"
    
    diff = actual_val - forecast_val
    color_hex = "#16A34A" if diff >= 0 else "#DC2626"
    
    if is_percentage:
        pct_val = diff / forecast_val
        return f'<span style="color: {color_hex}; font-weight: bold;">{pct_val:+.2%}</span>'
    else:
        return f'<span style="color: {color_hex}; font-weight: bold;">R {diff:+,.2f}</span>'

def get_html_ledger_table(years_filter, kpi):
    matrix_rows = []
    
    for yr in years_filter:
        year_total, year_lgb, year_sari = 0.0, 0.0, 0.0
        
        for qtr in [1, 2, 3, 4]:
            qtr_months = range((qtr-1)*3 + 1, qtr*3 + 1)
            qtr_act, qtr_lgb, qtr_sari = 0.0, 0.0, 0.0
            has_act = False
            
            for mo in qtr_months:
                dt = pd.Timestamp(year=yr, month=mo, day=1)
                m_name = dt.strftime('%B %Y')
                
                v_act = actuals.loc[dt, kpi] if dt in actuals.index and not pd.isna(actuals.loc[dt, kpi]) else np.nan
                v_lgb = lgb_df.loc[dt, kpi] if dt in lgb_df.index else np.nan
                v_sari = sari_df.loc[dt, kpi] if dt in sari_df.index else np.nan
                
                str_act = f"R {v_act:,.2f}" if not pd.isna(v_act) else "—"
                str_lgb = f"R {v_lgb:,.2f}" if not pd.isna(v_lgb) else "—"
                str_sari = f"R {v_sari:,.2f}" if not pd.isna(v_sari) else "—"
                
                lgb_v_str = color_variance_value(v_act, v_lgb, is_percentage=False)
                lgb_p_str = color_variance_value(v_act, v_lgb, is_percentage=True)
                sari_v_str = color_variance_value(v_act, v_sari, is_percentage=False)
                sari_p_str = color_variance_value(v_act, v_sari, is_percentage=True)
                
                matrix_rows.append([m_name, str_act, str_lgb, str_sari, lgb_v_str, lgb_p_str, sari_v_str, sari_p_str, "normal"])
                
                if not pd.isna(v_act):
                    qtr_act += v_act
                    has_act = True
                if not pd.isna(v_lgb): qtr_lgb += v_lgb
                if not pd.isna(v_sari): qtr_sari += v_sari
            
            # FIXED: Correct variable names used inside append tuple (p instead of pct)
            q_lgb_v = color_variance_value(qtr_act if has_act else np.nan, qtr_lgb, is_percentage=False)
            q_lgb_p = color_variance_value(qtr_act if has_act else np.nan, qtr_lgb, is_percentage=True)
            q_sari_v = color_variance_value(qtr_act if has_act else np.nan, qtr_sari, is_percentage=False)
            q_sari_p = color_variance_value(qtr_act if has_act else np.nan, qtr_sari, is_percentage=True)
            
            str_q_act = f"R {qtr_act:,.2f}" if has_act else "—"
            matrix_rows.append([
                f"TOTAL QUARTER {qtr} (Q{qtr})", str_q_act, f"R {qtr_lgb:,.2f}", f"R {qtr_sari:,.2f}",
                q_lgb_v, q_lgb_p, q_sari_v, q_sari_p, "subtotal"
            ])
            
            year_total += qtr_act
            year_lgb += qtr_lgb
            year_sari += qtr_sari
            
        # FIXED: Correct variable names used inside append tuple (p instead of pct)
        y_lgb_v = color_variance_value(year_total if year_total > 0 else np.nan, year_lgb, is_percentage=False)
        y_lgb_p = color_variance_value(year_total if year_total > 0 else np.nan, year_lgb, is_percentage=True)
        y_sari_v = color_variance_value(year_total if year_total > 0 else np.nan, year_sari, is_percentage=False)
        y_sari_p = color_variance_value(year_total if year_total > 0 else np.nan, year_sari, is_percentage=True)
        
        str_y_act = f"R {year_total:,.2f}" if year_total > 0 else "—"
        matrix_rows.append([
            f"🏆 GRAND TOTAL YEAR {yr}", str_y_act, f"R {year_lgb:,.2f}", f"R {year_sari:,.2f}",
            y_lgb_v, y_lgb_p, y_sari_v, y_sari_p, "grandtotal"
        ])
        
    html_output = \"\"\"
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; color: #0F172A; margin-top: 15px;">
        <thead>
            <tr style="border-bottom: 2px solid #0F172A; background-color: #F8FAFC;">
                <th style="padding: 12px; font-weight: bold; text-align: left;">Period Timeline</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">Observed Actuals</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">LightGBM Forecast</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">SARIMAX Forecast</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">LightGBM Variance</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">LGB Var %</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">SARIMAX Variance</th>
                <th style="padding: 12px; font-weight: bold; text-align: right;">SARIMAX Var %</th>
            </tr>
        </thead>
        <tbody>
    \"\"\"
    
    for row in matrix_rows:
        timeline, act, lgb_val, sari_val, l_v, l_p, s_v, s_p, row_type = row
        
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
            <td style="padding: 10px 12px; text-align: right;">{l_v}</td>
            <td style="padding: 10px 12px; text-align: right;">{l_p}</td>
            <td style="padding: 10px 12px; text-align: right;">{s_v}</td>
            <td style="padding: 10px 12px; text-align: right;">{s_p}</td>
        </tr>
        \"\"\"
        
    html_output += "</tbody></table>"
    return html_output

clean_html_table = get_html_ledger_table(plot_years, selected_kpi)
st.markdown(clean_html_table, unsafe_allow_html=True)
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(fixed_code)
print("✨ File system overwrite complete. Correct variables applied to disk.")
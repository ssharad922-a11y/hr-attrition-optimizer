import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

# Page Setup
st.set_page_config(page_title="Executive HR Retention Platform", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.pkl')
    calibrated_model = joblib.load('calibrated_model.pkl')
    lgbm_booster = joblib.load('lgbm_booster.pkl')
    test_df = pd.read_csv('test_data.csv', index_col=0)
    return preprocessor, calibrated_model, lgbm_booster, test_df

preprocessor, cal_lgb, lgbm_booster, test_df = load_artifacts()

# Mapping technical pipeline names to Plain English HR terms
FRIENDLY_NAMES = {
    'cat__OverTime_Yes': 'Frequent / Mandatory Overtime',
    'ord__StockOptionLevel': 'Stock Options / Equity Grant',
    'num__MonthlyIncome': 'Monthly Compensation',
    'num__Age': 'Employee Age',
    'ord__EnvironmentSatisfaction': 'Workplace Culture Satisfaction',
    'num__YearsAtCompany': 'Tenure at Company',
    'num__DistanceFromHome': 'Commute Distance from Home',
    'num__NumCompaniesWorked': 'Prior Career Moves / Job History',
    'num__StagnationRatio': 'Career Stagnation (Time Without Promotion)',
    'num__DailyRate': 'Daily Billing Rate',
    'num__TotalWorkingYears': 'Total Career Experience',
    'ord__JobInvolvement': 'Job Engagement / Involvement',
    'ord__WorkLifeBalance': 'Work-Life Balance Score',
    'num__ManagerTenureRatio': 'Managerial Continuity / Alignment',
    'cat__BusinessTravel_Travel_Frequently': 'Frequent Business Travel',
    'cat__MaritalStatus_Single': 'Marital Status: Single',
    'cat__JobRole_Sales Executive': 'Role: Sales Executive',
    'cat__JobRole_Research Scientist': 'Role: Research Scientist',
    'cat__JobRole_Laboratory Technician': 'Role: Laboratory Technician'
}

def clean_feature_label(name):
    if name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[name]
    clean = name.replace('cat__', '').replace('num__', '').replace('ord__', '')
    clean = clean.replace('_', ' ')
    return clean

# Application Header
st.title("🎯 People Analytics: Attrition Risk & Retention Engine")
st.markdown("Decision support system for identifying flight risks, diagnosing root causes, and allocating retention budgets.")

tab1, tab2 = st.tabs(["👤 Individual Employee Diagnostic", "💼 Executive Budget Allocation"])

feat_names = list(preprocessor.get_feature_names_out())
explainer = shap.TreeExplainer(lgbm_booster)

# ==========================================
# TAB 1: INDIVIDUAL DIAGNOSTIC
# ==========================================
with tab1:
    st.subheader("Employee Flight Risk Assessment")
    
    col_sel, col_stat = st.columns([1, 2])
    with col_sel:
        emp_id = st.selectbox("Select Employee:", options=test_df.index.tolist())
        emp_raw = test_df.loc[[emp_id]].drop(columns=['Attrition'])
        
        emp_proc = preprocessor.transform(emp_raw)
        cal_prob = cal_lgb.predict_proba(emp_proc)[0, 1]
        
    with col_stat:
        kpi1, kpi2, kpi3 = st.columns(3)
        risk_color = "red" if cal_prob >= 0.40 else "orange" if cal_prob >= 0.20 else "green"
        kpi1.metric("Flight Risk Level", f"{cal_prob:.1%}")
        
        annual_comp = emp_raw['MonthlyIncome'].values[0] * 12
        rep_cost = 1.5 * annual_comp
        kpi2.metric("Annual Compensation", f"${annual_comp:,.0f}")
        kpi3.metric("Estimated Replacement Cost", f"${rep_cost:,.0f}")

    # Plain English SHAP breakdown
    shap_vals = explainer(emp_proc)
    contributions = pd.DataFrame({
        'Feature': [clean_feature_label(f) for f in feat_names],
        'Impact': shap_vals.values[0]
    })
    
    # Separate into Retention Drivers and Flight Risks
    risks = contributions[contributions['Impact'] > 0.05].sort_values(by='Impact', ascending=True)
    anchors = contributions[contributions['Impact'] < -0.05].sort_values(by='Impact', ascending=False)
    
    st.markdown("---")
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.markdown("### 🔴 Key Flight Risk Catalysts *(Pushing them out)*")
        if len(risks) > 0:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(risks['Feature'], risks['Impact'], color='#FF4B4B')
            ax.set_xlabel("Relative Push Towards Leaving")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No significant negative risk catalysts identified.")
            
    with c_right:
        st.markdown("### 🟢 Key Retention Anchors *(Keeping them here)*")
        if len(anchors) > 0:
            fig2, ax2 = plt.subplots(figsize=(7, 3.5))
            ax2.barh(anchors['Feature'], np.abs(anchors['Impact']), color='#29B5E8')
            ax2.set_xlabel("Relative Holding Power")
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("No strong retention anchors identified.")

# ==========================================
# TAB 2: BUDGET OPTIMIZATION
# ==========================================
with tab2:
    st.subheader("Quarterly Retention Budget Optimizer")
    
    X_test_raw = test_df.drop(columns=['Attrition'])
    X_test_proc = preprocessor.transform(X_test_raw)
    probs = cal_lgb.predict_proba(X_test_proc)[:, 1]
    
    portfolio = X_test_raw.copy()
    portfolio['Flight_Risk'] = probs
    portfolio['Replacement_Cost'] = 1.5 * portfolio['MonthlyIncome'] * 12
    portfolio['Expected_Loss'] = portfolio['Flight_Risk'] * portfolio['Replacement_Cost']
    
    # Identify top SHAP driver for each employee
    test_shap = explainer(X_test_proc)
    top_driver_indices = np.argmax(test_shap.values, axis=1)
    portfolio['Top_Driver_Raw'] = [feat_names[i] for i in top_driver_indices]
    portfolio['Primary_Driver'] = portfolio['Top_Driver_Raw'].apply(clean_feature_label)
    
    def assign_action(row):
        driver = row['Top_Driver_Raw']
        salary = row['MonthlyIncome'] * 12
        if 'OverTime' in driver:
            action, cost, drop = "Workload Redistribution (Cap Overtime)", 500.0, 0.35
        elif 'MonthlyIncome' in driver or 'StockOption' in driver:
            action, cost, drop = "Equity / Salary Adjustment (+8%)", 0.08 * salary, 0.25
        elif 'Stagnation' in driver or 'Promotion' in driver:
            action, cost, drop = "Career Growth Mentorship Track", 2500.0, 0.20
        else:
            action, cost, drop = "Manager Stay Interview & Alignment", 300.0, 0.15
            
        new_risk = max(0.01, row['Flight_Risk'] - drop)
        savings = (row['Flight_Risk'] - new_risk) * row['Replacement_Cost']
        return pd.Series([action, cost, savings], index=['Action', 'Cost', 'Savings'])

    portfolio[['Recommended_Action', 'Action_Cost', 'Expected_Savings']] = portfolio.apply(assign_action, axis=1)
    
    candidate_pool = portfolio[portfolio['Flight_Risk'] >= 0.25].reset_index()

    b_col, kpi_box = st.columns([1, 2])
    with b_col:
        budget = st.slider("Total Available Budget ($):", min_value=5000, max_value=50000, value=25000, step=2500)
        
    n = len(candidate_pool)
    costs = candidate_pool['Action_Cost'].values
    savings = candidate_pool['Expected_Savings'].values
    
    c = -savings
    A = costs.reshape(1, -1)
    constraints = LinearConstraint(A, lb=0, ub=budget)
    integrality = np.ones(n)
    bounds = Bounds(0, 1)
    
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.success:
        selected_idx = np.where(res.x > 0.5)[0]
        funded = candidate_pool.iloc[selected_idx]
        
        spent = funded['Action_Cost'].sum()
        saved = funded['Expected_Savings'].sum()
        net = saved - spent
        roi = (net / spent) * 100 if spent > 0 else 0
        
        with kpi_box:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Interventions Funded", f"{len(funded)} / {n}")
            m2.metric("Budget Allocated", f"${spent:,.0f}")
            m3.metric("Turnover Loss Saved", f"${saved:,.0f}")
            m4.metric("Net ROI", f"{roi:,.0f}%")
            
        st.markdown("### Approved Strategic Retention Plan")
        disp_cols = ['index', 'Flight_Risk', 'Primary_Driver', 'Recommended_Action', 'Action_Cost', 'Expected_Savings']
        st.dataframe(funded[disp_cols].rename(columns={'index': 'Employee_ID'}).style.format({
            'Flight_Risk': '{:.1%}',
            'Action_Cost': '${:,.2f}',
            'Expected_Savings': '${:,.2f}'
        }), use_container_width=True)
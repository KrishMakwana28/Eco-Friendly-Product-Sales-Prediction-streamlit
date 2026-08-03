import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import date
import calendar

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Eco-Friendly Product Sales Prediction",
    page_icon="🌱",
    layout="wide"
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("eco_friendly_product_sales_model.pkl")

model = load_model()

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

h1{
    text-align:center;
    color:#2E8B57;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:12px;
    background-color:#2E8B57;
    color:white;
}

.stMetric{
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("🌱 Eco-Friendly Product Sales Prediction Dashboard")

st.markdown("""
### Predict Weekly Retail Sales

This dashboard predicts weekly retail sales using a trained **Linear Regression Pipeline**.

#### Model Features

- 🌍 Region
- 🌡 Average Temperature
- 📈 Google Trend Score
- 💰 Marketing Spend
- 🏪 Store Visits
- 🎉 Holiday Week
- 📅 Date

The application automatically extracts:

- 📆 Year
- 📅 Quarter
- 🗓 Day
- 📌 Month Number
""")

st.divider()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("📋 Input Parameters")

region = st.sidebar.selectbox(
    "🌍 Region",
    ["North", "South", "East", "West"]
)

selected_date = st.sidebar.date_input(
    "📅 Week Start Date",
    value=date.today()
)

avg_temp = st.sidebar.slider(
    "🌡 Average Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=25.0
)

google_trend_score = st.sidebar.slider(
    "📈 Google Trend Score",
    min_value=0,
    max_value=100,
    value=50
)

marketing_spend = st.sidebar.number_input(
    "💰 Marketing Spend ($)",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

store_visits = st.sidebar.number_input(
    "🏪 Store Visits",
    min_value=0,
    value=500,
    step=10
)

holiday = st.sidebar.selectbox(
    "🎉 Holiday Week",
    ["No", "Yes"]
)

holiday_flag = 1 if holiday == "Yes" else 0

# ------------------------------------------------
# DATE FEATURES
# ------------------------------------------------
year = selected_date.year
month_number = selected_date.month
day = selected_date.day
quarter = ((month_number - 1) // 3) + 1
month_name = calendar.month_name[month_number]

# ------------------------------------------------
# MODEL INPUT
# ------------------------------------------------
input_df = pd.DataFrame({

    "region":[region],
    "avg_temp":[avg_temp],
    "google_trend_score":[google_trend_score],
    "marketing_spend":[marketing_spend],
    "store_visits":[store_visits],
    "holiday_flag":[holiday_flag],
    "quarter":[quarter],
    "year":[year],
    "Day":[day],
    "month_number":[month_number]

})

# ------------------------------------------------
# DISPLAY INPUT
# ------------------------------------------------
st.subheader("📄 Input Summary")

display_df = pd.DataFrame({

    "Region":[region],
    "Date":[selected_date],
    "Day":[day],
    "Month":[month_name],
    "Quarter":[quarter],
    "Year":[year],
    "Holiday Week":[holiday],
    "Average Temperature":[avg_temp],
    "Google Trend Score":[google_trend_score],
    "Marketing Spend ($)":[marketing_spend],
    "Store Visits":[store_visits]

})

st.dataframe(display_df, use_container_width=True)

st.divider()
# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.button("🚀 Predict Sales", type="primary"):

    with st.spinner("Generating Prediction..."):

        prediction = model.predict(input_df)[0]

    weekly_sales = float(prediction)
    monthly_sales = weekly_sales * 4.33
    quarterly_sales = weekly_sales * 13
    yearly_sales = weekly_sales * 52

    st.success("✅ Sales Prediction Generated Successfully!")

    st.subheader("📊 Weekly Sales Prediction")

    max_sales = weekly_sales * 1.5 if weekly_sales > 0 else 100

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=weekly_sales,
            number={
                "prefix": "$ ",
                "valueformat": ",.2f"
            },
            title={
                "text": "<b>Predicted Weekly Sales</b>"
            },
            gauge={
                "axis": {
                    "range": [0, max_sales]
                },
                "bar": {
                    "color": "#2E8B57"
                },
                "steps": [
                    {
                        "range": [0, max_sales * 0.5],
                        "color": "#C8E6C9"
                    },
                    {
                        "range": [max_sales * 0.5, max_sales],
                        "color": "#81C784"
                    }
                ],
                "threshold": {
                    "line": {
                        "color": "red",
                        "width": 4
                    },
                    "thickness": 0.75,
                    "value": weekly_sales
                }
            }
        )
    )

    gauge.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("📈 Sales Forecast")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📅 Weekly",
            f"${weekly_sales:,.2f}"
        )

    with col2:
        st.metric(
            "🗓 Monthly",
            f"${monthly_sales:,.2f}"
        )

    with col3:
        st.metric(
            "📆 Quarterly",
            f"${quarterly_sales:,.2f}"
        )

    with col4:
        st.metric(
            "📊 Yearly",
            f"${yearly_sales:,.2f}"
        )

    st.divider()

    st.subheader("📋 Prediction Summary")

    summary = pd.DataFrame({
        "Metric": [
            "Weekly Sales",
            "Monthly Sales",
            "Quarterly Sales",
            "Yearly Sales"
        ],
        "Predicted Value ($)": [
            round(weekly_sales, 2),
            round(monthly_sales, 2),
            round(quarterly_sales, 2),
            round(yearly_sales, 2)
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.download_button(
        label="📥 Download Prediction",
        data=summary.to_csv(index=False),
        file_name="sales_prediction.csv",
        mime="text/csv"
    )

    st.info(
        """
        💡 **Model Information**

        - Algorithm: Linear Regression
        - Preprocessing: OneHotEncoder + StandardScaler
        - Prediction: Weekly Retail Sales
        """
    )

else:

    st.warning("👈 Enter the required values from the sidebar and click **Predict Sales**.")
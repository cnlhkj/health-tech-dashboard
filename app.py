import streamlit as st
import polars as pl
from google import genai

st.set_page_config(page_title="Churn Intervention AI", page_icon="🩺", layout="wide")
st.title("🩺 Health-Tech Churn Intervention")

@st.cache_data
def load_and_prep_data():
    try:
        pipeline = (
            pl.scan_csv("gym_data.csv")
            .with_row_index("user_id")
            .with_columns([
                (pl.lit("Member_") + pl.col("user_id").cast(pl.String)).alias("name"),
                # FEATURE ENGINEERING 2.0: Continuous Multi-Variate Risk Score
                # Base risk starts high (1.0). We subtract risk for healthy, continuous behaviors.
                (
                    1.0 
                    - (pl.col("Workout_Frequency (days/week)") * 0.12) # Major weight
                    - (pl.col("Session_Duration (hours)") * 0.05)      # Minor weight
                    + (pl.col("Age") * 0.002)                          # Slight continuous variance
                    - (pl.col("Calories_Burned") * 0.0001)             # High continuous variance
                )
                .clip(0.05, 0.95) # Cap the math so probabilities stay between 5% and 95%
                .alias("churn_probability"),
                (7.0 / (pl.col("Workout_Frequency (days/week)") + 0.1)).round(0).alias("days_since_last_log")
            ])
        )
        return pipeline.collect()
    
    except FileNotFoundError:
        st.error("⚠️ Please drag and drop 'gym_data.csv' into your Codespace first!")
        st.stop()

df = load_and_prep_data()

# ==========================================
# 📊 Visual Charts Section
# ==========================================
st.markdown("### 📊 Platform Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("**User Distribution by Workout Frequency**")
    # FIX 1: Bar chart showing the COUNT of users instead of overlapping dots
    distribution_data = (
        df.group_by("Workout_Frequency (days/week)")
        .len()
        .sort("Workout_Frequency (days/week)")
        .to_pandas()
        .set_index("Workout_Frequency (days/week)")
    )
    st.bar_chart(distribution_data)

with col2:
    st.write("**Average Calories Burned by Experience Level**")
    bar_data = (
        df.group_by("Experience_Level")
        .agg(pl.col("Calories_Burned").mean().round(0))
        .to_pandas().set_index("Experience_Level")
    )
    st.bar_chart(bar_data)

st.divider() 

# ==========================================
# ⚠️ Actionable Table and AI Intervention
# ==========================================

# FIX 2: Interactive Slider for the Product Manager!
st.subheader("⚙️ Intervention Settings")
threshold_percentage = st.slider(
    "Set Minimum Churn Risk Threshold for Intervention (%)", 
    min_value=50, max_value=95, value=70, step=5
)
# Convert percentage back to a decimal for Polars (e.g., 70 -> 0.70)
risk_threshold = threshold_percentage / 100.0

# Filter the dataframe dynamically based on the slider
high_risk_df = df.filter(pl.col("churn_probability") >= risk_threshold)

st.subheader(f"⚠️ High-Risk Members Flagged: {high_risk_df.height}")

# Only show the table and AI tool IF there are users who meet the threshold
if high_risk_df.height > 0:
    display_cols = ["name", "Age", "Experience_Level", "Workout_Frequency (days/week)", "churn_probability"]
    st.dataframe(high_risk_df.select(display_cols).to_pandas(), use_container_width=True) 

    selected_name = st.selectbox(
        "Select a member to draft an intervention for:", 
        high_risk_df["name"].to_list()
    )

    if st.button(f"Generate Retention Email for {selected_name}"):
        # Pull the key securely from Streamlit Secrets
        try:
            secure_api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            st.error("API Key not found! Please add it to Streamlit Secrets.")
            st.stop()
            
        user_data = high_risk_df.filter(pl.col("name") == selected_name).row(0, named=True)
            
        with st.spinner(f"Drafting personalized email for {selected_name}..."):
            client = genai.Client(api_key=secure_api_key)
            
            prompt = f"""
            Act as a Customer Success Manager for a premium health app. 
            Write a short, friendly, 3-sentence email to {user_data['name']}. 
            Acknowledge that they are at the '{user_data['Experience_Level']}' level, 
            but we noticed they are only logging {user_data['Workout_Frequency (days/week)']} days a week.
            Offer them a free 15-minute consultation to adjust their routine and get back on track.
            """
            
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            
            st.success("Draft Ready for Review!")
            st.text_area("Review and Edit Draft:", value=response.text, height=200)
else:
    st.success("No users found above this risk threshold! Try lowering the slider.")
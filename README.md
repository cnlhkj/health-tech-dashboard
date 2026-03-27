# 🩺 FitTech Retention Engine: Predictive Churn & AI Outreach

[FitTech Retention Engine: Predictive Churn & AI Outreach](https://health-tech-dashboard-hailyu.streamlit.app/)

## 📌 The Business Problem
In the highly competitive health and fitness tech space, acquiring a new user is significantly more expensive than retaining an existing one. Traditional analytics dashboards often fail because they are passive—they show who *has* churned, rather than providing actionable tools to intervene *before* it happens.

## 💡 The Solution
This application bridges the gap between predictive analytics and customer success operations. It ingests raw user behavioral data, calculates a dynamic churn risk score, and provides a sleek interface for stakeholders to instantly generate personalized, AI-driven retention outreach.

### Key Features:
* **Continuous Risk Scoring:** Moves beyond simple binary thresholds by using a multi-variate heuristic formula (incorporating workout frequency, session duration, age, and caloric burn) to assign a precise risk probability to every user.
* **Dynamic Stakeholder Controls:** Includes interactive UI elements (like risk threshold sliders) allowing Product Managers and Customer Success teams to filter cohorts based on current intervention budgets and bandwidth.
* **Generative Action (AI Integration):** Connects to the **Gemini 3.1 Flash Lite** model via API to auto-draft personalized, context-aware retention emails based on the specific user's behavioral drop-off.

## 🛠️ Technical Architecture & Engineering Choices

This project was built with a focus on production-readiness, memory efficiency, and rapid execution:

* **Data Pipeline (Polars):** Chosen over Pandas for its multi-threaded, lazy evaluation engine (`pl.scan_csv`). This ensures the app can scale to process massive datasets out-of-core without overwhelming server RAM.
* **Frontend (Streamlit):** Deployed a rapid, interactive web application to translate complex dataframes into a clean, stakeholder-friendly UI. 
* **LLM Integration (Google GenAI SDK):** Implemented targeted prompt engineering to transform quantitative user data into natural, empathetic text. API keys are managed securely via cloud environment variables.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cnlhkj/health-tech-dashboard.git
   cd health-tech-dashboard
2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt

3. **Set up your API Key:**
Create a .streamlit/secrets.toml file in the root directory and add your key:

    ```Ini, TOML
    GEMINI_API_KEY = "your_api_key_here"
4. **Launch the app:**

    ```Bash
    streamlit run app.py
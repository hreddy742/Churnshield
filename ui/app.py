"""Streamlit dashboard for predictions, model performance, and drift monitoring."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None


def _default_api_base_url() -> str:
    if Path("/.dockerenv").exists():
        return "http://backend:8000"
    return "http://127.0.0.1:8000"


API_BASE_URL = os.getenv("CHURNGUARD_API_BASE_URL", _default_api_base_url())


def _post_json(path: str, payload):
    if requests is None:
        raise RuntimeError("requests is required for the Streamlit dashboard")
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_json(path: str):
    if requests is None:
        raise RuntimeError("requests is required for the Streamlit dashboard")
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _render_predict_page() -> None:
    st.title("Predict")
    single_tab, batch_tab = st.tabs(["Single Customer", "Batch CSV Upload"])

    with single_tab:
        with st.form("predict_form"):
            tenure_months = st.number_input("Tenure Months", min_value=0.0, value=12.0)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=79.0)
            contract_type = st.selectbox("Contract Type", ["month-to-month", "one-year", "two-year"])
            num_support_tickets = st.number_input("Support Tickets", min_value=0, value=1)
            avg_satisfaction_score = st.slider("Satisfaction Score", min_value=0.0, max_value=10.0, value=7.0)
            customer_notes = st.text_area("Customer Notes", value="Customer reported a billing issue.")
            submitted = st.form_submit_button("Predict")

        if submitted:
            payload = [
                {
                    "tenure_months": tenure_months,
                    "monthly_charges": monthly_charges,
                    "contract_type": contract_type,
                    "num_support_tickets": num_support_tickets,
                    "avg_satisfaction_score": avg_satisfaction_score,
                    "customer_notes": customer_notes,
                }
            ]
            prediction = _post_json("/predict", payload)[0]
            explanation = _post_json("/explain", payload[0])

            st.metric("Churn Probability", f"{prediction['churn_probability']:.2%}")
            shap_df = pd.DataFrame(explanation["shap_values"])
            fig = px.bar(
                shap_df.sort_values("shap_value"),
                x="shap_value",
                y="feature",
                orientation="h",
                title="SHAP Waterfall View",
            )
            st.plotly_chart(fig, use_container_width=True)

    with batch_tab:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            results = _post_json("/predict", df.to_dict(orient="records"))
            st.dataframe(pd.DataFrame(results), use_container_width=True)


def _render_model_performance_page() -> None:
    st.title("Model Performance")
    performance = _get_json("/performance")
    history_df = pd.DataFrame(performance.get("history", []))
    if not history_df.empty:
        auc_fig = px.line(history_df, x="timestamp", y="auc", color="model_type", title="AUC History")
        st.plotly_chart(auc_fig, use_container_width=True)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No MLflow performance history available yet.")

    feature_csv = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble_feature_importance.csv")
    if os.path.exists(feature_csv):
        feature_df = pd.read_csv(feature_csv).head(20)
        importance_fig = px.bar(feature_df, x="mean_abs_shap", y="feature", orientation="h", title="Feature Importance")
        st.plotly_chart(importance_fig, use_container_width=True)


def _render_drift_page() -> None:
    st.title("Drift Monitor")
    drift_results = _get_json("/drift-report")
    if not drift_results:
        st.info("No drift report available yet.")
        return

    drift_df = pd.DataFrame(drift_results)
    drift_df["score"] = drift_df["psi"].fillna(drift_df["chi2_stat"])
    fig = px.bar(
        drift_df,
        x="feature",
        y="score",
        color="status",
        color_discrete_map={"no_drift": "green", "warning": "orange", "critical": "red"},
        title="Feature Drift Scores",
    )
    st.plotly_chart(fig, use_container_width=True)

    for _, row in drift_df.iterrows():
        color = {"no_drift": "green", "warning": "orange", "critical": "red"}[row["status"]]
        st.markdown(
            f"<span style='background-color:{color};padding:4px 8px;border-radius:6px;color:white'>{row['feature']}: {row['status']}</span>",
            unsafe_allow_html=True,
        )

    st.caption(f"Last checked: {datetime.now(timezone.utc).isoformat()}")


def main() -> None:
    """Render the Streamlit dashboard."""

    st.set_page_config(page_title="ChurnGuard", layout="wide")
    page = st.sidebar.radio("Navigation", ["Predict", "Model Performance", "Drift Monitor"])

    if page == "Predict":
        _render_predict_page()
    elif page == "Model Performance":
        _render_model_performance_page()
    else:
        _render_drift_page()


if __name__ == "__main__":
    main()

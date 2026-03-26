import numpy as np, pandas as pd
from faker import Faker
import uuid
from pathlib import Path

fake = Faker()
np.random.seed(42)

N = 50_000

NOTE_TEMPLATES = [
    "Very satisfied with the service",
    "Had billing issues last month, escalated to manager",
    "Long-time customer, loves the premium features",
    "Complained about slow support response times",
    "Asked about competitor pricing during last call",
    "Recently upgraded plan, happy with new features",
    "Mentioned cancellation on support call",
    "Referred two friends, strong advocate",
    "Issue with product quality, requested refund",
    "Happy with pricing, no complaints",
    "Frustrated with recent service outage",
    "Auto-renewed, no engagement in past 3 months",
    "Actively uses all features, power user",
    "Considering downgrade due to budget constraints",
    "Support ticket resolved quickly, satisfied",
    "Complained multiple times this quarter",
    "New customer, onboarding completed",
    "Left negative review, reached out by retention team",
    "Long-term customer, never had issues",
    "Reduced usage significantly in past 60 days",
]

# Realistic distributions
tenure = np.random.gamma(shape=2.2, scale=12, size=N).clip(1, 84).astype(int)
charges = np.random.normal(65, 20, N).clip(20, 120).round(2)
contract = np.random.choice(
    ["Month-to-Month", "One Year", "Two Year"],
    p=[0.55, 0.25, 0.20], size=N
)
tickets = np.random.poisson(2.1, N).clip(0, 15)
satisfaction = (np.random.beta(4, 2, N) * 4 + 1).round(1)
payment = np.random.choice(
    ["Electronic", "Mailed Check", "Bank Transfer", "Credit Card"],
    p=[0.35, 0.15, 0.25, 0.25], size=N
)
products = np.random.choice([1, 2, 3, 4], p=[0.40, 0.35, 0.18, 0.07], size=N)
notes = np.random.choice(NOTE_TEMPLATES, size=N)

# Churn probability (realistic correlations)
logit = (
    -0.30               # intercept to tune churn rate
    -0.04 * tenure
    + 0.02 * charges
    + 0.30 * tickets
    - 0.50 * satisfaction
    + np.where(contract == "Month-to-Month", 0.8,
      np.where(contract == "One Year", -0.2, -0.6))
    - 0.15 * products
    + np.random.normal(0, 0.4, N)  # noise
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

churn_prob = sigmoid(logit)
churn_label = (np.random.uniform(size=N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [str(uuid.uuid4()) for _ in range(N)],
    "tenure_months": tenure,
    "monthly_charges": charges,
    "contract_type": contract,
    "num_support_tickets": tickets,
    "avg_satisfaction_score": satisfaction,
    "payment_method": payment,
    "num_products": products,
    "customer_notes": notes,
    "churn_label": churn_label,
})

out_path = Path("ml/data/raw")
out_path.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path / "customer_data.csv", index=False)

churn_rate = df["churn_label"].mean()
print(f"Generated {N:,} records")
print(f"Churn rate: {churn_rate:.1%}")
print(f"Saved to ml/data/raw/customer_data.csv")
assert 0.20 <= churn_rate <= 0.30, f"Churn rate out of expected range: {churn_rate}"

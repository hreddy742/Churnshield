"""Generate a realistic synthetic customer churn dataset for ChurnGuard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    Faker = None


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "raw" / "customer_churn_synthetic.csv"
CONTRACT_TYPES = ("month-to-month", "one-year", "two-year")
SUPPORT_REASONS = (
    "billing issue",
    "slow support response",
    "service outage",
    "feature request",
    "pricing complaint",
    "login trouble",
    "setup assistance",
    "contract clarification",
)
POSITIVE_PHRASES = (
    "The service has been reliable lately.",
    "Recent support interactions were helpful and professional.",
    "The onboarding experience was smooth.",
    "The customer mentioned the portal is easy to use.",
    "They appreciate the current pricing plan.",
    "Network performance has improved over the last quarter.",
)
NEGATIVE_PHRASES = (
    "The customer is frustrated with repeated service interruptions.",
    "They reported unresolved billing disputes this month.",
    "The latest support case took too long to close.",
    "The customer is actively comparing competitors.",
    "They said the price feels too high for the value received.",
    "Recent conversations suggest growing dissatisfaction.",
)
NEUTRAL_PHRASES = (
    "The account owner requested a routine plan review.",
    "The customer asked for a usage summary before renewal.",
    "They contacted support for standard account maintenance.",
    "The team scheduled a check-in call for next month.",
    "There was a general inquiry about available add-ons.",
    "The customer left brief feedback without strong sentiment.",
)


@dataclass(frozen=True)
class SyntheticDataConfig:
    n_rows: int = 50_000
    seed: int = 42
    output_path: Path = DEFAULT_OUTPUT_PATH


class _FallbackFaker:
    """Small fallback used when Faker is unavailable in the runtime."""

    WORDS = (
        "account",
        "customer",
        "support",
        "renewal",
        "service",
        "billing",
        "review",
        "contract",
        "usage",
        "feedback",
        "experience",
        "team",
        "stability",
        "pricing",
        "followup",
    )

    @staticmethod
    def seed(_: int) -> None:
        return None

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def sentence(self, nb_words: int = 8) -> str:
        words = self._rng.choice(self.WORDS, size=nb_words, replace=True)
        sentence = " ".join(words)
        return f"{sentence.capitalize()}."


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _sample_contract_types(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    return rng.choice(CONTRACT_TYPES, size=n_rows, p=[0.58, 0.24, 0.18])


def _sample_tenure_months(rng: np.random.Generator, contract_types: np.ndarray) -> np.ndarray:
    raw_tenure = rng.gamma(shape=2.2, scale=12.0, size=contract_types.shape[0])
    tenure = np.clip(np.rint(raw_tenure), 1, 72).astype(int)

    one_year_mask = contract_types == "one-year"
    two_year_mask = contract_types == "two-year"
    tenure[one_year_mask] = np.maximum(tenure[one_year_mask], rng.integers(10, 25, size=one_year_mask.sum()))
    tenure[two_year_mask] = np.maximum(tenure[two_year_mask], rng.integers(20, 49, size=two_year_mask.sum()))
    return tenure


def _sample_monthly_charges(
    rng: np.random.Generator, contract_types: np.ndarray, tenure_months: np.ndarray
) -> np.ndarray:
    base_by_contract = np.select(
        [contract_types == "month-to-month", contract_types == "one-year", contract_types == "two-year"],
        [88.0, 76.0, 69.0],
        default=80.0,
    )
    charges = rng.normal(loc=base_by_contract, scale=14.0)
    loyalty_discount = np.clip(tenure_months * 0.18, 0, 9)
    charges = np.clip(charges - loyalty_discount, 18.0, 140.0)
    return np.round(charges, 2)


def _sample_support_tickets(
    rng: np.random.Generator, contract_types: np.ndarray, churn_propensity: np.ndarray
) -> np.ndarray:
    lam = np.select(
        [contract_types == "month-to-month", contract_types == "one-year", contract_types == "two-year"],
        [2.4, 1.6, 1.1],
        default=1.7,
    )
    lam = lam + np.clip(churn_propensity, 0, 1.2)
    return np.clip(rng.poisson(lam=lam), 0, 12).astype(int)


def _sample_satisfaction_scores(
    rng: np.random.Generator, support_tickets: np.ndarray, monthly_charges: np.ndarray
) -> np.ndarray:
    score = rng.normal(loc=7.3, scale=1.15, size=support_tickets.shape[0])
    score -= support_tickets * 0.34
    score -= np.clip((monthly_charges - 90.0) / 40.0, 0, None)
    score += rng.normal(loc=0.0, scale=0.35, size=support_tickets.shape[0])
    return np.round(np.clip(score, 1.0, 10.0), 2)


def _build_customer_notes(
    faker: Faker,
    rng: np.random.Generator,
    churn_score: np.ndarray,
    support_tickets: np.ndarray,
) -> list[str]:
    notes: list[str] = []
    for score, tickets in zip(churn_score, support_tickets):
        if score > 0.7:
            sentiment_pool = NEGATIVE_PHRASES
        elif score < 0.35:
            sentiment_pool = POSITIVE_PHRASES
        else:
            sentiment_pool = NEUTRAL_PHRASES

        snippets = [
            rng.choice(sentiment_pool),
            f"The customer opened {tickets} support ticket(s) in the recent review window.",
        ]
        if rng.random() < 0.35:
            snippets.append(f"Internal follow-up note: {faker.sentence(nb_words=8)}")
        if rng.random() < 0.2:
            snippets.append(f"Primary issue category: {rng.choice(SUPPORT_REASONS)}.")
        notes.append(" ".join(snippets))
    return notes


def generate_synthetic_customer_data(
    n_rows: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic churn dataset with realistic feature interactions."""

    rng = np.random.default_rng(seed)
    if Faker is None:
        faker = _FallbackFaker(rng)
    else:
        faker = Faker()
        Faker.seed(seed)

    contract_types = _sample_contract_types(rng, n_rows)
    tenure_months = _sample_tenure_months(rng, contract_types)

    initial_propensity = (
        (contract_types == "month-to-month") * 0.65
        + (contract_types == "one-year") * 0.2
        + np.clip((18 - tenure_months) / 24.0, 0, 1.0) * 0.5
        + rng.normal(0.0, 0.12, size=n_rows)
    )
    monthly_charges = _sample_monthly_charges(rng, contract_types, tenure_months)
    num_support_tickets = _sample_support_tickets(rng, contract_types, initial_propensity)
    avg_satisfaction_score = _sample_satisfaction_scores(rng, num_support_tickets, monthly_charges)

    linear_score = (
        -1.28
        + 1.1 * (contract_types == "month-to-month")
        + 0.32 * (contract_types == "one-year")
        + 0.022 * np.maximum(monthly_charges - 82.0, 0)
        + 0.24 * num_support_tickets
        - 0.06 * np.minimum(tenure_months, 24)
        - 0.72 * ((avg_satisfaction_score - 5.0) / 2.5)
        + rng.normal(0.0, 0.48, size=n_rows)
    )
    churn_probability = _sigmoid(linear_score)
    churn_label = rng.binomial(1, churn_probability).astype(int)

    customer_notes = _build_customer_notes(faker, rng, churn_probability, num_support_tickets)
    customer_ids = [f"CUST-{idx:06d}" for idx in range(1, n_rows + 1)]

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "contract_type": contract_types,
            "num_support_tickets": num_support_tickets,
            "avg_satisfaction_score": avg_satisfaction_score,
            "customer_notes": customer_notes,
            "churn_label": churn_label,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000, help="Number of synthetic customer rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticDataConfig(n_rows=args.rows, seed=args.seed, output_path=args.output)
    df = generate_synthetic_customer_data(n_rows=config.n_rows, seed=config.seed)
    save_path = save_dataset(df, config.output_path)
    churn_rate = df["churn_label"].mean()
    print(f"Saved {len(df):,} rows to {save_path}")
    print(f"Observed churn rate: {churn_rate:.2%}")


if __name__ == "__main__":
    main()

"""Readable and technical reporting for an agricultural-macro run.

The reporting layer deliberately consumes ordinary pandas objects and mappings.
It does not know how a model was fitted, which keeps model measurement separate
from narrative generation and makes every statement traceable to an input table.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REGIME_LABELS = {
    "weather_shock": "Weather / Crop Supply Shock",
    "food_security": "Inventory / Food Security Stress",
    "policy_stress": "Government / Trade Policy Amplification",
    "input_expansion": "Agricultural Input Expansion",
    "food_inflation": "Food Inflation Transmission",
    "monetary_tightening": "Monetary Tightening / Stagflation",
    "supply_response": "Supply Response / Mean Reversion",
}

CAUSAL_CHAINS = {
    "weather_shock": [
        "Measured: weather, crop-condition, and yield inputs produced the leading score",
        "Model pathway: persistent adverse weather could weaken yields",
        "Model pathway: lower yields could reduce production and exportable supply",
        "Model pathway: tighter supply could increase the grain risk premium",
        "Hypothesis to monitor: policy and food-inflation risks may follow",
    ],
    "food_security": [
        "Measured: exporter-inventory and scarcity inputs produced the leading score",
        "Model pathway: tighter exporter stocks could reduce exportable surplus",
        "Model pathway: less tradable supply could intensify importer competition",
        "Model pathway: physical grain pricing could strengthen",
        "Hypothesis to monitor: food-security policy risk may increase",
    ],
    "policy_stress": [
        "Measured: the aligned policy-stress input produced the leading score",
        "Model pathway: intervention could alter tradable supply",
        "Model pathway: altered tradable supply could affect international prices",
        "Model pathway: import costs and food-security pressure could change",
        "Hypothesis to monitor: secondary inflation risk may change",
    ],
    "input_expansion": [
        "Measured: crop/input and margin inputs produced the leading score",
        "Model pathway: better affordability could strengthen farmer margins",
        "Model pathway: stronger margins could support acreage and input demand",
        "Model pathway: demand and feedstock costs could affect input producers",
        "Hypothesis to monitor: a next-season supply response may follow",
    ],
    "food_inflation": [
        "Measured: grain, energy, and inflation inputs produced the leading score",
        "Model pathway: sustained input prices could raise food-production costs",
        "Model pathway: higher food costs could broaden headline inflation",
        "Model pathway: broader inflation could reprice expectations",
        "Hypothesis to monitor: rates markets may absorb greater policy risk",
    ],
    "monetary_tightening": [
        "Measured: rates, dollar, and inflation inputs produced the leading score",
        "Model pathway: persistent inflation could lift policy-rate expectations",
        "Model pathway: policy repricing could strengthen front-end yields and the dollar",
        "Model pathway: higher rates and a stronger dollar could tighten financial conditions",
        "Hypothesis to monitor: demand and growth expectations may weaken",
    ],
    "supply_response": [
        "Measured: supply-response inputs produced the leading score",
        "Model pathway: strong crop economics could improve planting incentives",
        "Model pathway: better incentives could expand input use or acreage",
        "Model pathway: favorable yields could rebuild production and inventories",
        "Hypothesis to monitor: greater supply may create mean-reversion pressure",
    ],
}


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        return None if isinstance(item, float) and not np.isfinite(item) else item
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Series):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        frame = value.reset_index()
        return [_plain(row) for row in frame.to_dict(orient="records")]
    if pd.isna(value) if np.isscalar(value) and not isinstance(value, str) else False:
        return None
    return value


def _regime_key(label: Any) -> str:
    text = str(label).strip().lower().replace("/", " ").replace("-", " ")
    text = "_".join(text.split())
    aliases = {
        "weather_crop_supply_shock": "weather_shock",
        "weather_supply_shock": "weather_shock",
        "inventory_food_security_stress": "food_security",
        "food_security_stress": "food_security",
        "government_trade_policy_amplification": "policy_stress",
        "government_intervention": "policy_stress",
        "agricultural_input_expansion": "input_expansion",
        "fertilizer_expansion": "input_expansion",
        "food_inflation_transmission": "food_inflation",
        "monetary_tightening_stagflation": "monetary_tightening",
        "supply_response_mean_reversion": "supply_response",
    }
    return aliases.get(text, text)


def _normalize_regime_columns(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out.columns = [_regime_key(c) for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    return out.sort_index()


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        frame = value if isinstance(value.index, pd.RangeIndex) else value.reset_index()
        return frame.to_dict(orient="records")
    if isinstance(value, Mapping):
        return [dict(value)]
    result: list[dict[str, Any]] = []
    for item in value:
        if is_dataclass(item):
            result.append(asdict(item))
        elif isinstance(item, Mapping):
            result.append(dict(item))
        else:
            result.append(vars(item))
    return result


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}{suffix}"


def _trend(change: float, threshold: float = 1.0) -> str:
    if change > threshold:
        return "Strengthening ↑"
    if change < -threshold:
        return "Weakening ↓"
    return "Stable →"


def _table_html(frame: pd.DataFrame, *, index: bool = False) -> str:
    if frame is None or frame.empty:
        return '<p class="muted">No observations available.</p>'
    view = frame.copy()
    for column in view.select_dtypes(include=["number"]).columns:
        view[column] = view[column].map(lambda x: "—" if pd.isna(x) else f"{x:,.3g}")
    return view.to_html(index=index, border=0, classes="data-table", escape=True)


def _transition_vector(
    transition_probabilities: Any,
    dominant: str,
    horizon: str = "1m",
) -> pd.Series:
    """Extract one next-state probability vector from supported transition APIs."""

    if transition_probabilities is None:
        return pd.Series(dtype=float)
    if hasattr(transition_probabilities, "latest_for"):
        try:
            return pd.to_numeric(transition_probabilities.latest_for(horizon), errors="coerce")
        except (KeyError, IndexError, ValueError):
            pass
    if hasattr(transition_probabilities, "latest_probabilities"):
        latest = transition_probabilities.latest_probabilities
        if isinstance(latest, pd.DataFrame) and not latest.empty:
            if horizon in latest.index:
                return pd.to_numeric(latest.loc[horizon], errors="coerce")
            return pd.to_numeric(latest.iloc[0], errors="coerce")
    if isinstance(transition_probabilities, Mapping):
        preferred = transition_probabilities.get(horizon)
        if preferred is None:
            preferred = transition_probabilities.get("1_month")
        if preferred is None and transition_probabilities:
            preferred = next(iter(transition_probabilities.values()))
        transition_probabilities = preferred
    if isinstance(transition_probabilities, pd.DataFrame):
        frame = transition_probabilities.copy()
        if {"horizon", "next_regime", "probability"}.issubset(frame.columns):
            selected = frame.loc[frame["horizon"].astype(str) == horizon]
            if selected.empty:
                selected = frame
            return pd.to_numeric(
                selected.drop_duplicates("next_regime", keep="last").set_index("next_regime")["probability"],
                errors="coerce",
            )
        if dominant in frame.index:
            return pd.to_numeric(frame.loc[dominant], errors="coerce")
        if horizon in frame.index:
            return pd.to_numeric(frame.loc[horizon], errors="coerce")
        if len(frame):
            return pd.to_numeric(frame.iloc[-1], errors="coerce")
    if isinstance(transition_probabilities, pd.Series):
        return pd.to_numeric(transition_probabilities, errors="coerce")
    if isinstance(transition_probabilities, Mapping):
        return pd.Series(transition_probabilities, dtype=float)
    return pd.Series(dtype=float)


def _latest_transition(transition_probabilities: Any, dominant: str) -> tuple[str, float]:
    row = _transition_vector(transition_probabilities, dominant)
    row.index = [_regime_key(c) for c in row.index]
    row = row.dropna()
    if row.empty:
        return dominant, float("nan")
    next_regime = str(row.idxmax())
    probability = float(row.max())
    # The public reporting contract uses decimal probabilities in [0, 1].
    # Accept legacy percentage inputs, but normalize once at this boundary.
    if probability > 1.000001:
        probability /= 100.0
    return next_regime, float(np.clip(probability, 0.0, 1.0))


def _top_contributions(contributions: Any, regime: str, limit: int = 5) -> pd.DataFrame:
    if contributions is None:
        return pd.DataFrame(columns=["factor", "contribution"])
    if hasattr(contributions, "latest_contributions"):
        contributions = contributions.latest_contributions
    if isinstance(contributions, Mapping):
        selected = contributions.get(regime, contributions)
        if isinstance(selected, Mapping):
            out = pd.DataFrame(
                [(str(k), float(v)) for k, v in selected.items()],
                columns=["factor", "contribution"],
            )
        else:
            out = pd.DataFrame(selected)
    else:
        out = pd.DataFrame(contributions).copy()
    if out.empty:
        return pd.DataFrame(columns=["factor", "contribution"])
    out.columns = [str(c).lower() for c in out.columns]
    if "regime" in out.columns:
        out = out[out["regime"].map(_regime_key) == regime]
    factor_col = next((c for c in ("factor", "feature", "driver") if c in out), None)
    value_col = next(
        (
            c
            for c in (
                "score_point_contribution",
                "contribution",
                "raw_contribution",
                "value",
                "points",
            )
            if c in out
        ),
        None,
    )
    if factor_col is None or value_col is None:
        return pd.DataFrame(columns=["factor", "contribution"])
    result = out[[factor_col, value_col]].rename(
        columns={factor_col: "factor", value_col: "contribution"}
    )
    result["contribution"] = pd.to_numeric(result["contribution"], errors="coerce")
    result = result.dropna().assign(abs_value=lambda x: x["contribution"].abs())
    return result.sort_values("abs_value", ascending=False).drop(columns="abs_value").head(limit)


def _top_contribution_change(
    contributions: Any,
    regime: str,
    latest_date: Any,
    prior_date: Any,
) -> pd.Series | None:
    """Return the largest measured contribution change over a score window.

    This deliberately uses the historical contribution panel rather than the
    latest contribution ranking.  A large current contribution is not evidence
    that it caused the score to move during the requested window.
    """

    if contributions is None:
        return None
    source = (
        contributions.contributions
        if hasattr(contributions, "contributions")
        else contributions
    )
    if isinstance(source, Mapping):
        return None
    frame = pd.DataFrame(source).copy()
    if frame.empty:
        return None
    frame.columns = [str(column).lower() for column in frame.columns]
    factor_col = next(
        (column for column in ("factor", "feature", "driver") if column in frame),
        None,
    )
    value_col = next(
        (
            column
            for column in (
                "score_point_contribution",
                "contribution",
                "raw_contribution",
                "value",
                "points",
            )
            if column in frame
        ),
        None,
    )
    if not {"date", "regime"}.issubset(frame.columns) or factor_col is None or value_col is None:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame[
        (frame["regime"].map(_regime_key) == _regime_key(regime))
        & frame["date"].notna()
        & frame[value_col].notna()
    ]
    if frame.empty:
        return None

    latest_candidates = frame.loc[frame["date"] <= pd.Timestamp(latest_date), "date"]
    prior_candidates = frame.loc[frame["date"] <= pd.Timestamp(prior_date), "date"]
    if latest_candidates.empty or prior_candidates.empty:
        return None
    selected_latest = latest_candidates.max()
    selected_prior = prior_candidates.max()
    latest_values = (
        frame.loc[frame["date"] == selected_latest]
        .groupby(factor_col, sort=False)[value_col]
        .sum()
    )
    prior_values = (
        frame.loc[frame["date"] == selected_prior]
        .groupby(factor_col, sort=False)[value_col]
        .sum()
    )
    changes = latest_values.subtract(prior_values, fill_value=0.0).dropna()
    if changes.empty:
        return None
    factor = changes.abs().idxmax()
    return pd.Series(
        {"factor": str(factor), "change": float(changes.loc[factor])}
    )


def _what_changed(scores: pd.DataFrame, contributions: Any, lookback: int = 5) -> list[str]:
    if scores.empty:
        return ["No regime history was available for change detection."]
    prior_pos = max(0, len(scores) - 1 - lookback)
    delta = (scores.iloc[-1] - scores.iloc[prior_pos]).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    statements: list[str] = []
    for regime, value in delta.head(5).items():
        old = float(scores.iloc[prior_pos][regime])
        new = float(scores.iloc[-1][regime])
        contribution_change = _top_contribution_change(
            contributions,
            str(regime),
            scores.index[-1],
            scores.index[prior_pos],
        )
        attribution = ""
        if contribution_change is not None:
            factor = str(contribution_change["factor"]).replace("_", " ")
            points = float(contribution_change["change"])
            attribution = (
                f"; largest measured contribution change: {factor} "
                f"({points:+.1f} score points)"
            )
        else:
            driver = _top_contributions(contributions, str(regime), 1)
            if not driver.empty:
                factor = str(driver.iloc[0]["factor"]).replace("_", " ")
                attribution = f"; largest current measured contribution: {factor}"
        statements.append(
            f"{REGIME_LABELS.get(str(regime), str(regime).replace('_', ' ').title())} "
            f"moved from {old:.1f} to {new:.1f} ({value:+.1f}){attribution}."
        )
    return statements


def _signal_frame(signals: Any) -> pd.DataFrame:
    records = _records(signals)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    aliases = {
        "instrument": "asset",
        "score": "signal",
        "conviction": "confidence",
        "horizon": "expected_horizon",
        "driver": "economic_driver",
        "invalidation": "invalidation_condition",
    }
    rename: dict[Any, str] = {}
    existing = {str(c).lower() for c in frame.columns}
    for column in frame.columns:
        canonical = str(column).lower()
        target = aliases.get(canonical, canonical)
        if target in existing and target != canonical:
            target = canonical
        rename[column] = target
    frame = frame.rename(columns=rename)
    if "date" in frame.columns:
        parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
        if parsed_dates.notna().any():
            frame = frame.loc[parsed_dates == parsed_dates.max()].copy()
    if "strength" in frame:
        frame["strength"] = pd.to_numeric(frame["strength"], errors="coerce")
    elif "signal" in frame:
        frame["strength"] = pd.to_numeric(frame["signal"], errors="coerce").abs()
    if "confidence" in frame:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
        if frame["confidence"].dropna().max() <= 1.000001:
            frame["confidence"] *= 100.0
    if "confidence" in frame:
        frame = frame.sort_values("confidence", ascending=False)
    elif "strength" in frame:
        frame = frame.sort_values("strength", key=lambda x: x.abs(), ascending=False)
    if "asset" in frame.columns:
        frame = frame.drop_duplicates(subset=["asset"], keep="first")
    return frame.reset_index(drop=True)


def _quality_mapping(data_quality: Any) -> dict[str, Any]:
    if data_quality is None:
        return {"score": 0.0, "status": "UNKNOWN", "issues": ["No quality report supplied"]}
    if is_dataclass(data_quality):
        return asdict(data_quality)
    if isinstance(data_quality, Mapping):
        return dict(data_quality)
    return dict(vars(data_quality))


def _market_inferences(
    factors: pd.DataFrame,
    dominant: str,
    next_regime: str,
    next_probability: float,
) -> list[str]:
    """Produce at most three explicitly labelled, measurement-backed inferences."""

    if factors is None or factors.empty:
        return ["Model inference: no cross-market inference passed the available-data test."]
    latest = pd.to_numeric(factors.iloc[-1], errors="coerce")
    normalized = {_regime_key(k): float(v) for k, v in latest.dropna().items()}
    candidates: list[tuple[float, str]] = []
    scarcity = normalized.get("physical_scarcity", normalized.get("f10_physical_scarcity", np.nan))
    if np.isfinite(scarcity):
        candidates.append(
            (
                abs(scarcity),
                "Model inference: wheat performance relative to dollar pressure suggests the physical signal may differ from the macro signal."
                if scarcity > 0
                else "Model inference: dollar-adjusted wheat behavior does not currently confirm a physical scarcity premium.",
            )
        )
    affordability = normalized.get("ag_affordability", normalized.get("corn_urea", np.nan))
    if np.isfinite(affordability):
        candidates.append(
            (
                abs(affordability),
                "Model inference: improving crop-to-fertilizer affordability could support input demand before it appears in company results."
                if affordability > 0
                else "Model inference: weak input affordability is a counterweight to bullish fertilizer-demand narratives.",
            )
        )
    inventory = normalized.get("exporter_stocks_to_use", np.nan)
    if np.isfinite(inventory):
        candidates.append(
            (
                abs(inventory),
                "Model inference: exporter inventory tightness may matter more for international prices than comfortable aggregate world stocks."
                if inventory < 0
                else "Model inference: exporter inventory conditions do not currently support a scarcity premium.",
            )
        )
    if next_regime != dominant and np.isfinite(next_probability):
        candidates.append(
            (
                next_probability,
                f"Model inference: the transition model favors {REGIME_LABELS.get(next_regime, next_regime)} over persistence, so current-regime positioning may be late-cycle.",
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in candidates[:3]] or [
        "Model inference: no cross-market inference passed the available-data test."
    ]


def create_charts(
    output_dir: str | Path,
    market: pd.DataFrame,
    factors: pd.DataFrame,
    regime_scores: pd.DataFrame,
    transition_probabilities: Any,
    signals: Any,
    backtest: Any = None,
    *,
    as_of: Any = None,
    source_label: str = "Configured sources",
) -> list[Path]:
    """Create the ten standard report charts, skipping only unavailable inputs."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_dir = Path(output_dir) / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    market = market.sort_index().copy()
    factors = factors.sort_index().copy() if factors is not None else pd.DataFrame(index=market.index)
    scores = _normalize_regime_columns(regime_scores)
    latest_date = pd.Timestamp(market.index.max() if len(market.index) else as_of).date().isoformat()
    paths: list[Path] = []

    def finish(fig: Any, ax: Any, name: str, title: str, ylabel: str = "") -> None:
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(
            0.01,
            0.01,
            f"Source: {source_label} | Decision panel through: {latest_date} | "
            "Series-specific observation dates and ages: diagnostics/series_metadata.csv",
            fontsize=7,
            color="#667085",
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        path = chart_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)

    if not scores.empty:
        fig, ax = plt.subplots(figsize=(12, 5.4))
        scores.tail(504).rename(columns=REGIME_LABELS).plot(ax=ax, linewidth=1.35)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, ncol=1, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        finish(fig, ax, "01_regime_scores.png", "Regime Scores Over Time", "Score (0–100)")

    wheat_dxy = (
        market[["wheat", "dxy"]].tail(504).dropna(how="all")
        if {"wheat", "dxy"}.issubset(market.columns)
        else pd.DataFrame()
    )
    if not wheat_dxy.empty and wheat_dxy.notna().any().all():
        normalized = wheat_dxy
        normalized = normalized / normalized.apply(lambda c: c.dropna().iloc[0]) * 100
        fig, ax = plt.subplots(figsize=(10, 4.6))
        normalized.rename(columns={"wheat": "Wheat", "dxy": "DXY"}).plot(ax=ax)
        ax.legend(frameon=False)
        finish(fig, ax, "02_wheat_vs_dxy.png", "Wheat vs DXY", "Indexed to 100")

    ratio = None
    if {"corn", "urea"}.issubset(market.columns):
        ratio = market["corn"] / market["urea"].replace(0, np.nan)
    elif "corn_urea" in factors:
        ratio = factors["corn_urea"]
    if ratio is not None:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ratio.tail(504).plot(ax=ax, color="#15803d")
        finish(fig, ax, "03_corn_urea.png", "Corn / Urea Fertilizer Affordability", "Ratio")

    if "exporter_stocks_to_use" in market:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        market["exporter_stocks_to_use"].tail(756).plot(ax=ax, color="#b45309")
        finish(fig, ax, "04_exporter_stocks_to_use.png", "Exporter Stocks-to-Use", "Percent")

    macro_cols = [c for c in ("wheat", "us_2y", "breakeven_5y") if c in market]
    if len(macro_cols) >= 2:
        macro = market[macro_cols].tail(504).copy()
        rolling_mean = macro.rolling(252, min_periods=20).mean()
        rolling_std = macro.rolling(252, min_periods=20).std(ddof=0).replace(0, np.nan)
        macro = (macro - rolling_mean) / rolling_std
        fig, ax = plt.subplots(figsize=(10, 4.6))
        macro.rename(columns={"wheat": "Wheat", "us_2y": "US 2Y", "breakeven_5y": "5Y breakeven"}).plot(ax=ax)
        ax.legend(frameon=False)
        finish(
            fig,
            ax,
            "05_grain_rates_breakeven.png",
            "Wheat, Front-End Rates, and Breakevens",
            "Causal 252-session rolling z-score",
        )

    margin = None
    for name in ("nitrogen_margin", "nitrogen_margin_proxy", "FertilizerMargin"):
        if name in factors:
            margin = factors[name]
            break
    if margin is None and {"urea", "natural_gas"}.issubset(market):
        margin = market["urea"] - 2.0 * market["natural_gas"]
    if margin is not None:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        margin.tail(504).plot(ax=ax, color="#7c3aed")
        finish(fig, ax, "06_fertilizer_margin.png", "Nitrogen Producer Margin Proxy", "Proxy units")

    equity = None
    drawdown = None
    if backtest is not None:
        equity = getattr(backtest, "equity_curve", None)
        drawdown = getattr(backtest, "drawdown", None)
        if isinstance(backtest, Mapping):
            equity = backtest.get("equity_curve", equity)
            drawdown = backtest.get("drawdown", drawdown)
    if equity is not None and len(equity):
        initial_value = getattr(equity, "attrs", {}).get("initial_value")
        equity = pd.Series(equity)
        first_valid = equity.dropna().iloc[0] if not equity.dropna().empty else np.nan
        baseline = (
            float(initial_value)
            if initial_value is not None and np.isfinite(float(initial_value))
            else first_valid
        )
        display_equity = equity / baseline if np.isfinite(baseline) and baseline != 0 else equity
        fig, ax = plt.subplots(figsize=(10, 4.2))
        display_equity.plot(ax=ax, color="#0f766e")
        finish(fig, ax, "07_portfolio_equity.png", "Strategy Equity Curve", "Growth of 1")
        if drawdown is None:
            drawdown = equity / equity.cummax() - 1.0
    if drawdown is not None and len(drawdown):
        drawdown = pd.Series(drawdown)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.fill_between(drawdown.index, 100 * drawdown.values, 0, color="#dc2626", alpha=0.55)
        finish(fig, ax, "08_portfolio_drawdown.png", "Simulated Strategy Drawdown", "Percent")

    signal_frame = _signal_frame(signals)
    if not signal_frame.empty and {"asset", "signal"}.issubset(signal_frame):
        heat = signal_frame.dropna(subset=["signal"]).head(15)
        if not heat.empty:
            fig, ax = plt.subplots(figsize=(8, max(2.8, 0.34 * len(heat))))
            image = ax.imshow(
                heat[["signal"]].to_numpy(),
                cmap="RdYlGn",
                vmin=-1,
                vmax=1,
                aspect="auto",
            )
            ax.set_yticks(range(len(heat)), labels=heat["asset"].astype(str))
            ax.set_xticks([0], labels=["Signal"])
            for i, value in enumerate(heat["signal"]):
                ax.text(0, i, f"{value:+.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
            finish(fig, ax, "09_signal_heatmap.png", "Current Signal Heatmap")

    next_regime, _ = _latest_transition(transition_probabilities, str(scores.iloc[-1].idxmax()) if not scores.empty else "")
    dominant_regime = str(scores.iloc[-1].idxmax()) if not scores.empty else ""
    row = _transition_vector(transition_probabilities, dominant_regime).dropna()
    if not row.empty:
        if row.max() <= 1.000001:
            row *= 100
        row.index = [REGIME_LABELS.get(_regime_key(c), str(c)) for c in row.index]
        row = row.sort_values()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        row.plot.barh(ax=ax, color=["#0f766e" if _regime_key(c) == next_regime else "#94a3b8" for c in row.index])
        ax.set_xlim(0, max(100, row.max() * 1.1))
        finish(fig, ax, "10_transition_probabilities.png", "One-Month Regime Transition Probabilities", "Probability (%)")

    return paths


def generate_reports(
    output_dir: str | Path,
    *,
    as_of: Any,
    system_version: str,
    market: pd.DataFrame,
    factors: pd.DataFrame,
    regime_scores: pd.DataFrame,
    contributions: Any,
    transition_probabilities: Any,
    signals: Any,
    portfolio_weights: pd.Series | Mapping[str, float] | None,
    risk_report: Mapping[str, Any] | None,
    backtest: Any,
    scenarios: pd.DataFrame | None,
    data_quality: Any,
    source_label: str,
    run_metadata: Mapping[str, Any] | None = None,
    catalysts: pd.DataFrame | None = None,
    benchmarks: pd.DataFrame | None = None,
    stress_tests: pd.DataFrame | None = None,
    model_health: Mapping[str, Any] | None = None,
    lead_lag: Any = None,
    research_questions: pd.DataFrame | None = None,
    series_metadata: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Generate an executive HTML report, Markdown report, and machine JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores = _normalize_regime_columns(regime_scores)
    if scores.empty:
        raise ValueError("regime_scores must contain at least one scored date")
    latest = scores.iloc[-1]
    dominant = str(latest.idxmax())
    dominant_label = REGIME_LABELS.get(dominant, dominant.replace("_", " ").title())
    one_day = float(latest[dominant] - scores.iloc[-2][dominant]) if len(scores) > 1 else 0.0
    prior_week = scores.iloc[max(0, len(scores) - 6)]
    one_week = float(latest[dominant] - prior_week[dominant])
    next_regime, next_probability = _latest_transition(transition_probabilities, dominant)
    next_label = REGIME_LABELS.get(next_regime, next_regime.replace("_", " ").title())
    quality = _quality_mapping(data_quality)
    quality_score = float(quality.get("score", quality.get("overall_score", 0.0)) or 0.0)
    quality_multiplier = float(
        quality.get("confidence_multiplier", np.clip(quality_score / 100.0, 0.0, 1.0))
    )
    quality_status = str(quality.get("status", "unknown")).lower()
    signal_frame = _signal_frame(signals)
    health = dict(model_health or {})
    stability = float(np.clip(health.get("regime_stability", 0.5), 0.0, 1.0))
    signal_confidence = 0.0
    if "confidence" in signal_frame and not signal_frame["confidence"].dropna().empty:
        signal_confidence = float(
            np.clip(signal_frame["confidence"].dropna().median() / 100.0, 0.0, 1.0)
        )
    activation = float(np.clip(float(latest[dominant]) / 100.0, 0.0, 1.0))
    bounded_quality = float(np.clip(quality_multiplier, 0.0, 1.0))
    confidence = 100.0 * bounded_quality * (
        0.40 * activation
        + 0.25 * signal_confidence
        + 0.20 * stability
        + 0.15 * bounded_quality
    )
    demo_fixture = "demo" in source_label.lower() or "synthetic" in source_label.lower()
    if demo_fixture:
        # Demo outcomes cannot calibrate historical reliability, crowding, or
        # event risk. Keep their reported conviction visibly bounded.
        confidence = min(confidence, 75.0)
    confidence = float(np.clip(confidence, 0.0, 100.0))
    confidence_method = (
        "uncalibrated heuristic conviction from regime activation, current signal "
        "confidence, recent regime stability, and data quality; not a probability"
    )
    changes = _what_changed(scores, contributions)
    market_inferences = _market_inferences(factors, dominant, next_regime, next_probability)
    top_drivers = _top_contributions(contributions, dominant, 5)
    if portfolio_weights is None:
        portfolio = pd.Series(dtype=float)
    elif isinstance(portfolio_weights, pd.Series):
        portfolio = pd.to_numeric(portfolio_weights, errors="coerce").dropna().copy()
    else:
        portfolio = pd.Series(dict(portfolio_weights), dtype=float)
    risk = dict(risk_report or {})
    metric_map: dict[str, Any] = {}
    if backtest is not None:
        metrics = getattr(backtest, "metrics", None)
        if metrics is None and isinstance(backtest, Mapping):
            metrics = backtest.get("metrics", {})
        if is_dataclass(metrics):
            metrics = asdict(metrics)
        if isinstance(metrics, pd.Series):
            metrics = metrics.to_dict()
        if isinstance(metrics, Mapping):
            metric_map = dict(metrics)

    chart_paths = create_charts(
        output,
        market,
        factors,
        scores,
        transition_probabilities,
        signal_frame,
        backtest,
        as_of=as_of,
        source_label=source_label,
    )

    regime_rows = []
    prior_day = scores.iloc[-2] if len(scores) > 1 else latest
    for regime, value in latest.sort_values(ascending=False).items():
        delta_day = float(value - prior_day[regime])
        delta_week = float(value - prior_week[regime])
        regime_rows.append(
            {
                "Regime": REGIME_LABELS.get(str(regime), str(regime)),
                "Score": float(value),
                "Δ 1D": delta_day,
                "Δ 1W": delta_week,
                "Trend": _trend(delta_week),
            }
        )
    regime_table = pd.DataFrame(regime_rows)

    chain = CAUSAL_CHAINS.get(dominant, ["Measured shock", "Fundamental transmission", "Market repricing"])
    policy_method_text = (
        "The policy-amplification score uses the point-in-time aligned policy_stress "
        "series. In demo mode that series is a deterministic fixture; event-weighted "
        "aggregation is used only when structured policy-event records are explicitly supplied. "
        f"Current policy-amplification score: {_fmt(latest.get('policy_stress'))}/100."
    )
    trades_eligible = quality_status != "failed" and quality_multiplier > 0.0
    trade_signal_frame = signal_frame if trades_eligible else signal_frame.iloc[0:0]
    minimum_signal = float((run_metadata or {}).get("trade_min_signal", 0.05))
    confidence_floor = float(
        (run_metadata or {}).get("trade_confidence_floor", 0.0)
    )
    if not trade_signal_frame.empty:
        executable = pd.Series(True, index=trade_signal_frame.index)
        if "direction" in trade_signal_frame:
            executable &= trade_signal_frame["direction"].astype(str).str.lower().ne("flat")
        if "signal" in trade_signal_frame:
            executable &= pd.to_numeric(
                trade_signal_frame["signal"], errors="coerce"
            ).abs().ge(minimum_signal)
        elif "strength" in trade_signal_frame:
            executable &= pd.to_numeric(
                trade_signal_frame["strength"], errors="coerce"
            ).abs().ge(minimum_signal)
        if "confidence" in trade_signal_frame:
            confidence_values = pd.to_numeric(
                trade_signal_frame["confidence"], errors="coerce"
            )
            executable &= confidence_values.gt(max(0.0, 100.0 * confidence_floor))
        trade_signal_frame = trade_signal_frame.loc[executable]
    trade_columns = [
        c
        for c in ("asset", "direction", "strength", "confidence", "expected_horizon", "economic_driver")
        if c in trade_signal_frame
    ]
    trade_table = (
        trade_signal_frame[trade_columns].head(5).copy()
        if trade_columns
        else pd.DataFrame()
    )
    if not trade_table.empty:
        trade_table.insert(0, "rank", range(1, len(trade_table) + 1))
    trade_details: list[str] = []
    for row in trade_signal_frame.head(3).to_dict(orient="records"):
        asset = str(row.get("asset", "instrument"))
        direction = str(row.get("direction", "flat"))
        driver = str(row.get("economic_driver", "measured factor agreement"))
        invalidation = str(
            row.get(
                "invalidation_condition",
                "the signed signal or its supporting regime evidence reverses",
            )
        )
        trade_details.append(
            f"{asset} ({direction}) — Thesis: {driver}. Confirmation: the signed signal "
            f"and regime evidence persist. Invalidation: {invalidation}. Main risk: "
            "the measured relationship may be sample-specific or already priced."
        )
    issues = quality.get("issues", []) or []
    issue_strings = []
    for issue in issues:
        if isinstance(issue, Mapping):
            issue_strings.append(str(issue.get("message") or issue.get("code") or issue))
        else:
            issue_strings.append(str(issue))

    driver_name = "measured factor agreement"
    if not top_drivers.empty:
        driver_name = str(top_drivers.iloc[0]["factor"]).replace("_", " ")
    next_probability_text = _fmt(next_probability * 100.0, 1, "%")
    executive = [
        f"The dominant measured regime is {dominant_label}, with a score of {latest[dominant]:.1f}/100 and {confidence:.0f}% heuristic conviction after data-quality adjustment; this is not a calibrated probability.",
        f"It changed {one_week:+.1f} points over the latest five-session window; the largest current contribution is {driver_name}.",
        f"The one-month transition model assigns its highest probability to {next_label} ({next_probability_text}).",
        (
            "Trade candidates are suppressed because data quality failed."
            if not trades_eligible
            else "Trade candidates are ranked from standardized signals, regime agreement, and data quality; they are research outputs rather than execution instructions."
        ),
        f"The most important model risk is {'; '.join(issue_strings[:1]) if issue_strings else 'a reversal in the leading measured drivers and regime transition evidence'}.",
    ]

    strongest_for = []
    strongest_against = []
    for row in top_drivers.itertuples(index=False):
        statement = f"{str(row.factor).replace('_', ' ').title()}: {float(row.contribution):+.2f} score contribution."
        (strongest_for if float(row.contribution) >= 0 else strongest_against).append(statement)
    if not strongest_for:
        strongest_for = ["The dominant score is the highest of the seven measured regimes."]
    if not strongest_against:
        competing = latest.drop(labels=[dominant]).sort_values(ascending=False)
        if not competing.empty:
            strongest_against = [
                f"Competing regime: {REGIME_LABELS.get(str(competing.index[0]), str(competing.index[0]))} scored {competing.iloc[0]:.1f}/100."
            ]
        else:
            strongest_against = ["No quantified counter-signal was available."]

    invalidations: list[str] = []
    if not signal_frame.empty and "invalidation_condition" in signal_frame:
        invalidations = list(
            dict.fromkeys(str(v) for v in signal_frame["invalidation_condition"].dropna())
        )[:3]
    if not invalidations:
        invalidations = [
            "The dominant regime score falls below 50 or loses leadership.",
            "Its leading factor contributions reverse for two consecutive updates.",
            "The expected next-regime probability falls below 40% or another transition becomes more likely.",
        ]

    physical_cols = [c for c in ("weather_stress", "crop_condition", "yield_revision", "exporter_stocks_to_use", "wheat", "corn", "soybeans") if c in market]
    physical_snapshot = market[physical_cols].tail(2).T if physical_cols else pd.DataFrame()
    if not physical_snapshot.empty:
        physical_snapshot.columns = [str(pd.Timestamp(c).date()) for c in physical_snapshot.columns]
    macro_cols = [c for c in ("dxy", "us_2y", "breakeven_5y", "natural_gas", "urea") if c in market]
    macro_snapshot = market[macro_cols].tail(2).T if macro_cols else pd.DataFrame()
    if not macro_snapshot.empty:
        macro_snapshot.columns = [str(pd.Timestamp(c).date()) for c in macro_snapshot.columns]
    freshness_table = pd.DataFrame()
    if series_metadata is not None and not series_metadata.empty:
        freshness_table = series_metadata.copy()
        if "series_id" not in freshness_table.columns:
            freshness_table = freshness_table.reset_index()
        freshness_columns = [
            column
            for column in (
                "series_id",
                "source",
                "unit",
                "last_observation_date",
                "last_available_date",
                "data_age_days",
                "max_staleness_days",
                "stale",
            )
            if column in freshness_table
        ]
        freshness_table = freshness_table[freshness_columns]

    scenario_table = scenarios if scenarios is not None else pd.DataFrame()
    catalyst_table = catalysts.copy() if catalysts is not None else pd.DataFrame()
    benchmark_table = benchmarks.reset_index() if benchmarks is not None and not benchmarks.empty else pd.DataFrame()
    stress_table = stress_tests.reset_index() if stress_tests is not None and not stress_tests.empty else pd.DataFrame()
    lead_lag_table = getattr(lead_lag, "granger_tests", pd.DataFrame())
    lead_lag_disclaimer = str(
        getattr(
            lead_lag,
            "disclaimer",
            "Lead/lag statistics measure predictive association, not structural causality.",
        )
    )
    research_question_table = (
        research_questions.copy() if research_questions is not None else pd.DataFrame()
    )
    weight_table = portfolio.rename("Weight").to_frame().reset_index(names="Asset") if not portfolio.empty else pd.DataFrame()
    scalar_risk = {
        key: value
        for key, value in risk.items()
        if np.isscalar(value)
        and not isinstance(value, (dict, list, tuple, pd.Series, pd.DataFrame))
    }
    risk_table = pd.DataFrame([scalar_risk]) if scalar_risk else pd.DataFrame()
    metric_table = pd.DataFrame([metric_map]) if metric_map else pd.DataFrame()

    cross_rows: list[dict[str, Any]] = []
    if factors is not None and not factors.empty:
        latest_factors = pd.to_numeric(factors.iloc[-1], errors="coerce")
        cross_definitions = {
            "physical_scarcity": ("Wheat vs DXY", "positive supports physical scarcity; negative is non-confirming"),
            "corn_urea": ("Corn / Urea", "positive means improving fertilizer affordability"),
            "nitrogen_margin": ("Nitrogen margin", "positive means stronger product/feedstock economics"),
            "food_inflation": ("Grain + breakeven", "positive means stronger inflation transmission"),
            "dxy_momentum": ("DXY trend", "positive is a macro headwind to dollar-priced grain"),
        }
        for key, (label, interpretation) in cross_definitions.items():
            if key in latest_factors and pd.notna(latest_factors[key]):
                cross_rows.append(
                    {
                        "Signal": label,
                        "Value": float(latest_factors[key]),
                        "Interpretation": interpretation,
                    }
                )
    cross_table = pd.DataFrame(cross_rows)

    benchmark_counterevidence: str | None = None
    if not benchmark_table.empty and {"benchmark", "sharpe_ratio"}.issubset(benchmark_table.columns):
        strategy_row = benchmark_table.loc[benchmark_table["benchmark"].astype(str) == "strategy"]
        alternatives = benchmark_table.loc[benchmark_table["benchmark"].astype(str) != "strategy"]
        if not strategy_row.empty and not alternatives.empty:
            best = alternatives.sort_values("sharpe_ratio", ascending=False).iloc[0]
            strategy_sharpe = float(strategy_row.iloc[0]["sharpe_ratio"])
            if float(best["sharpe_ratio"]) > strategy_sharpe:
                benchmark_counterevidence = (
                    f"Simple benchmark {best['benchmark']} achieved Sharpe {float(best['sharpe_ratio']):.2f} "
                    f"versus {strategy_sharpe:.2f} for the regime strategy in this sample."
                )
                strongest_against.insert(
                    0,
                    benchmark_counterevidence,
                )
                strongest_against = strongest_against[:3]
                executive.append(benchmark_counterevidence)

    payload = {
        "date": str(pd.Timestamp(as_of).date()),
        "system_version": system_version,
        "source_label": source_label,
        "dominant_regime": dominant,
        "dominant_regime_label": dominant_label,
        "score": float(latest[dominant]),
        "confidence": confidence,
        "confidence_method": confidence_method,
        "change_1d": one_day,
        "change_1w": one_week,
        "trend": _trend(one_week),
        "expected_next_regime": next_regime,
        "expected_next_regime_label": next_label,
        "expected_next_probability": next_probability,
        "expected_next_probability_percent": next_probability * 100.0,
        "executive_summary": executive,
        "regime_scores": regime_rows,
        "current_causal_chain": chain,
        "what_changed": changes,
        "what_the_market_may_be_missing": market_inferences,
        "strongest_evidence_for": strongest_for[:3],
        "strongest_evidence_against": strongest_against[:3],
        "invalidations": invalidations,
        "signals": _plain(trade_table),
        "trade_details": trade_details,
        "portfolio_weights": _plain(portfolio),
        "risk": _plain(risk),
        "backtest_metrics": _plain(metric_map),
        "scenarios": _plain(scenario_table),
        "upcoming_catalysts": _plain(catalyst_table),
        "benchmarks": _plain(benchmark_table),
        "historical_stress_tests": _plain(stress_table),
        "model_health": _plain(health),
        "lead_lag_granger_tests": _plain(lead_lag_table),
        "lead_lag_disclaimer": lead_lag_disclaimer,
        "research_question_status": _plain(research_question_table),
        "series_freshness": _plain(freshness_table),
        "data_quality": _plain(quality),
        "charts": [str(p.relative_to(output)) for p in chart_paths],
        "run_metadata": _plain(run_metadata or {}),
    }
    payload = _plain(payload)
    json_path = output / "monitor.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    def ordered(items: Sequence[str]) -> str:
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))

    if not trades_eligible:
        quality_notice = (
            "DATA QUALITY FAILED — trade candidates and trade rationales are suppressed; "
            "regime measurements remain visible for diagnosis only."
        )
    elif quality_status == "warning":
        quality_notice = (
            "DATA QUALITY WARNING — confidence is haircutted; review the quality issues "
            "before using any research candidate."
        )
    elif demo_fixture:
        quality_notice = (
            "DEMO FIXTURE — conviction is an uncalibrated research heuristic and is "
            "capped at 75%; it is not an empirical success probability."
        )
    else:
        quality_notice = ""

    markdown_lines = [
        "# AGRICULTURAL MACRO REGIME MONITOR",
        "",
        f"**Date:** {payload['date']}  ",
        f"**System Version:** {system_version}  ",
        f"**Data:** {source_label}",
        "",
        f"**{quality_notice}**" if quality_notice else "",
        "" if quality_notice else "",
        "## Current Dominant Regime",
        "",
        f"**{dominant_label}** — {latest[dominant]:.1f}/100, heuristic conviction {confidence:.0f}% (uncalibrated; not a probability), one-week change {one_week:+.1f}, {_trend(one_week)}.",
        "",
        f"Expected next regime: **{next_label}** ({next_probability_text}).",
        "",
        "## Executive Summary",
        "",
        " ".join(executive),
        "",
        "## Regime Dashboard",
        "",
        regime_table.to_markdown(index=False, floatfmt=".1f"),
        "",
        "## Current Causal Chain",
        "",
        " → ".join(chain),
        "",
        "## What Changed",
        "",
        ordered(changes),
        "",
        "## Best Research Trades",
        "",
        trade_table.to_markdown(index=False) if not trade_table.empty else "No qualified trade candidates.",
        "",
        ordered(trade_details) if trade_details else "No trade rationale is available.",
        "",
        "## What the Market May Be Missing",
        "",
        ordered(market_inferences),
        "",
        "## Physical Market",
        "",
        "Aligned decision-panel values below may be carried from an earlier release; see the series-freshness table for actual observation and availability dates.",
        "",
        physical_snapshot.to_markdown() if not physical_snapshot.empty else "No physical-market snapshot available.",
        "",
        "### Series Freshness",
        "",
        freshness_table.to_markdown(index=False) if not freshness_table.empty else "No series-freshness metadata was supplied.",
        "",
        "## Government Policy",
        "",
        policy_method_text,
        "",
        "## Farmer & Fertilizer Economics",
        "",
        "Crop/input affordability and feedstock-adjusted fertilizer margins are measured separately; product-price inflation is not treated as producer profit.",
        "",
        "## Macro Transmission",
        "",
        macro_snapshot.to_markdown() if not macro_snapshot.empty else "No macro snapshot available.",
        "",
        "## Cross-Market Signals",
        "",
        cross_table.to_markdown(index=False) if not cross_table.empty else "No qualified cross-market signals.",
        "",
        "## Strongest Evidence For the Thesis",
        "",
        ordered(strongest_for[:3]),
        "",
        "## Strongest Evidence Against the Thesis",
        "",
        ordered(strongest_against[:3]),
        "",
        "## Invalidation",
        "",
        ordered(invalidations),
        "",
        "## Portfolio & Risk",
        "",
        weight_table.to_markdown(index=False) if not weight_table.empty else "No active portfolio weights.",
        "",
        risk_table.to_markdown(index=False) if not risk_table.empty else "No scalar risk summary available.",
        "",
        "## Scenario Analysis",
        "",
        scenario_table.to_markdown(index=False) if not scenario_table.empty else "No scenario estimates available.",
        "",
        "## Benchmark Comparison",
        "",
        benchmark_table.to_markdown(index=False) if not benchmark_table.empty else "No benchmark results available.",
        "",
        "## Historical Stress Tests",
        "",
        stress_table.to_markdown(index=False) if not stress_table.empty else "No overlapping stress periods available.",
        "",
        "## Upcoming Catalysts",
        "",
        catalyst_table.to_markdown(index=False) if not catalyst_table.empty else "No verified catalysts are loaded for this horizon.",
        "",
        "## Data Quality",
        "",
        f"Status: **{quality.get('status', 'UNKNOWN')}**; score: **{quality_score:.1f}/100**.",
    ]
    if issue_strings:
        markdown_lines.extend(["", ordered(issue_strings[:10])])
    markdown_lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            (
                f"The system locates the cycle in {dominant_label}, led by {driver_name}. "
                f"The highest-probability next state is {next_label}. "
                + (
                    "Trade candidates are suppressed because final data quality failed; regime outputs are diagnostic only. "
                    if not trades_eligible
                    else "Qualified ranked signals express that measurement with conservative risk controls. "
                )
                + "The view is invalidated if the leading score and its factor contributions reverse. All outputs are research candidates, not automatic orders."
            ),
            "",
            "## Technical Appendix",
            "",
            "See `monitor.json` and the CSV artifacts for exact factor values, contributions, transitions, weights, risk, and reproducibility metadata.",
        ]
    )
    markdown_path = output / "report.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    transition_table = _transition_vector(
        transition_probabilities, dominant
    ).rename("Probability").to_frame()
    technical_lines = [
        "# AGRICULTURAL MACRO REGIME MONITOR — TECHNICAL LAYER",
        "",
        f"As of {payload['date']} · System {system_version} · {source_label}",
        "",
        "## Regime Scores and Changes",
        "",
        regime_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Dominant-Regime Contributions",
        "",
        top_drivers.to_markdown(index=False) if not top_drivers.empty else "No contribution rows.",
        "",
        "## One-Month Transition Probabilities",
        "",
        transition_table.to_markdown() if not transition_table.empty else "No transition vector.",
        "",
        "## Current Signals",
        "",
        signal_frame.to_markdown(index=False) if not signal_frame.empty else "No signal rows.",
        "",
        "## Current Weights",
        "",
        weight_table.to_markdown(index=False) if not weight_table.empty else "No weights.",
        "",
        "## Risk",
        "",
        risk_table.to_markdown(index=False) if not risk_table.empty else "No scalar risk summary.",
        "",
        "## Backtest Metrics",
        "",
        metric_table.to_markdown(index=False) if not metric_table.empty else "No backtest metrics.",
        "",
        "## Benchmark Comparison",
        "",
        benchmark_table.to_markdown(index=False) if not benchmark_table.empty else "No benchmarks.",
        "",
        "## Scenario and Historical Stress Results",
        "",
        scenario_table.to_markdown(index=False) if not scenario_table.empty else "No scenarios.",
        "",
        stress_table.to_markdown(index=False) if not stress_table.empty else "No overlapping historical stress windows.",
        "",
        "## Lead/Lag Research Checks",
        "",
        lead_lag_disclaimer,
        "",
        lead_lag_table.to_markdown(index=False) if not lead_lag_table.empty else "No lead/lag tests were run.",
        "",
        "## Master Research-Question Status",
        "",
        research_question_table.to_markdown(index=False) if not research_question_table.empty else "No research-question audit was produced.",
        "",
        "## Data Quality and Model Health",
        "",
        f"Data quality: {quality.get('status', 'UNKNOWN')} ({quality_score:.2f}/100). Model health: {health.get('status', 'unknown')}.",
        "",
        "Exact full-history factors, contributions, transition histories, positions, costs, drift, and provenance are stored in the adjacent CSV/JSON artifacts.",
    ]
    technical_path = output / "technical_report.md"
    technical_path.write_text("\n".join(technical_lines) + "\n", encoding="utf-8")

    hero_cards = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in (
            ("Dominant regime", dominant_label),
            ("Score", f"{latest[dominant]:.1f} / 100"),
            ("Heuristic conviction", f"{confidence:.0f}%"),
            ("One-week change", f"{one_week:+.1f}"),
            ("Expected next", next_label),
            ("Transition probability", next_probability_text),
        )
    )
    chart_html = "".join(
        f'<figure><img src="{escape(str(path.relative_to(output)))}" alt="{escape(path.stem)}"><figcaption>{escape(path.stem.replace("_", " ").title())}</figcaption></figure>'
        for path in chart_paths
    )
    quality_banner = (
        f'<div class="warning"><strong>{escape(quality_notice)}</strong></div>'
        if quality_notice
        else ""
    )
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agricultural Macro Regime Monitor — {escape(payload['date'])}</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#dce3ec;--paper:#fff;--wash:#f4f7fa;--accent:#0f766e;--warn:#b45309}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:auto;padding:34px 22px 70px}} h1{{font-size:30px;margin:0}} h2{{margin-top:38px;padding-bottom:8px;border-bottom:1px solid var(--line)}} h3{{margin-top:24px}}
.eyebrow{{color:var(--accent);font-weight:700;letter-spacing:.08em;font-size:12px}} .meta,.muted{{color:var(--muted)}}
.hero{{background:linear-gradient(135deg,#ffffff,#eaf5f3);border:1px solid var(--line);border-radius:16px;padding:28px;box-shadow:0 8px 28px #26364a10}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:22px}} .card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px}}
.card span{{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}} .card strong{{font-size:17px}}
.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}}
.data-table{{width:100%;border-collapse:collapse;font-size:13px}} .data-table th{{text-align:left;background:#eef3f7}} .data-table th,.data-table td{{padding:9px;border-bottom:1px solid var(--line)}}
.chain{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}} .chain span{{background:#e6f2f0;border:1px solid #bddbd6;border-radius:8px;padding:9px 12px}} .chain b{{color:var(--accent)}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}} figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}} figure img{{width:100%;height:auto}} figcaption{{color:var(--muted);font-size:12px}}
.warning{{border-left:4px solid var(--warn);background:#fff8e7;padding:12px 15px}} ol{{padding-left:22px}} code{{background:#eef3f7;padding:2px 5px;border-radius:4px}}
@media(max-width:600px){{.charts{{grid-template-columns:1fr}}main{{padding:20px 12px}}}}
</style></head><body><main>
<section class="hero"><div class="eyebrow">AGRICULTURAL MACRO REGIME MONITOR</div><h1>{escape(payload['date'])}</h1>
<p class="meta">System {escape(system_version)} · {escape(source_label)} · Research output, not automatic execution</p><div class="cards">{hero_cards}</div></section>
{quality_banner}
<h2>Executive Summary</h2><div class="panel"><p>{escape(' '.join(executive))}</p></div>
<h2>Regime Dashboard</h2><div class="panel">{_table_html(regime_table)}</div>
<h2>Current Causal Chain</h2><div class="panel chain">{'<b>→</b>'.join(f'<span>{escape(item)}</span>' for item in chain)}</div>
<h2>What Changed</h2><div class="panel"><ol>{''.join(f'<li>{escape(item)}</li>' for item in changes)}</ol></div>
<h2>Physical Market</h2><div class="panel"><p>Values are aligned decision-panel observations and may be carried from an earlier release. Actual observation, availability, and age fields are shown below.</p>{_table_html(physical_snapshot, index=True)}{_table_html(freshness_table)}</div>
<h2>Government Policy</h2><div class="panel"><p>{escape(policy_method_text)}</p></div>
<h2>Farmer & Fertilizer Economics</h2><div class="panel"><p>Affordability and producer-margin signals are computed from crop/input ratios and feedstock-adjusted fertilizer prices. The measurement enters the input-expansion score and trade ranking; it is not treated as a direct profit forecast.</p></div>
<h2>Macro Transmission</h2><div class="panel">{_table_html(macro_snapshot, index=True)}</div>
<h2>Cross-Market Signals</h2><div class="panel">{_table_html(cross_table)}</div>
<h2>Best Research Trades</h2><div class="panel">{_table_html(trade_table)}<ol>{''.join(f'<li>{escape(item)}</li>' for item in trade_details)}</ol></div>
<h2>What the Market May Be Missing</h2><div class="panel"><ol>{''.join(f'<li>{escape(item)}</li>' for item in market_inferences)}</ol></div>
<h2>Portfolio & Risk</h2><div class="panel">{_table_html(weight_table)}{_table_html(risk_table)}{_table_html(metric_table)}</div>
<h2>Scenario Analysis</h2><div class="panel">{_table_html(scenario_table)}</div>
<h2>Benchmark Comparison</h2><div class="panel">{_table_html(benchmark_table)}</div>
<h2>Historical Stress Tests</h2><div class="panel">{_table_html(stress_table)}</div>
<h2>Strongest Evidence For</h2><div class="panel"><ol>{''.join(f'<li>{escape(item)}</li>' for item in strongest_for[:3])}</ol></div>
<h2>Strongest Evidence Against</h2><div class="panel"><ol>{''.join(f'<li>{escape(item)}</li>' for item in strongest_against[:3])}</ol></div>
<h2>Invalidation</h2><div class="panel"><ol>{''.join(f'<li>{escape(item)}</li>' for item in invalidations)}</ol></div>
<h2>Upcoming Catalysts</h2><div class="panel">{_table_html(catalyst_table)}</div>
<h2>Data Quality</h2><div class="panel {'warning' if issue_strings else ''}"><p>Status: <strong>{escape(str(quality.get('status', 'UNKNOWN')))}</strong> · score {quality_score:.1f}/100.</p><ol>{''.join(f'<li>{escape(item)}</li>' for item in issue_strings[:10])}</ol></div>
<h2>Model Health</h2><div class="panel"><p>Status: <strong>{escape(str(health.get('status', 'unknown')))}</strong> · recent regime stability {_fmt(100 * float(health.get('regime_stability', np.nan)), 1, '%')}.</p><ol>{''.join(f'<li>{escape(str(item))}</li>' for item in health.get('warnings', []))}</ol></div>
<h2>Charts</h2><div class="charts">{chart_html}</div>
<h2>Technical Appendix</h2><div class="panel"><p>Exact machine-readable inputs and results are saved in <code>monitor.json</code> and CSV artifacts alongside this report. The manifest records the code version, configuration hash, data snapshot hash, timestamp, and fixed seed.</p>{_table_html(top_drivers)}</div>
</main></body></html>"""
    html_path = output / "report.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return {
        "html": html_path,
        "markdown": markdown_path,
        "technical_markdown": technical_path,
        "json": json_path,
        "charts": chart_paths,
    }

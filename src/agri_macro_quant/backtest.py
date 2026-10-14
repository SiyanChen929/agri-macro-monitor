"""Lagged, transaction-cost-aware portfolio backtesting and validation."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .portfolio import DEFAULT_ASSET_ALIASES, construct_portfolio_history
from .risk import drawdown_risk_multiplier, drawdown_series


DEFAULT_BACKTEST_CONFIG: dict[str, Any] = {
    "annualization": 252,
    "execution_lag": 1,
    "transaction_cost_bps": 2.0,
    "slippage_bps": 1.0,
    "commission_bps": 0.0,
    "futures_roll_cost_bps": 1.0,
    "amortize_roll_cost": True,
    "rebalance_frequency": "daily",
    "initial_capital": 1.0,
    "risk_free_rate": 0.0,
    "var_confidence": 0.95,
    "apply_drawdown_control": True,
}


def _backtest_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    outer = dict(config or {})
    nested = dict(outer.get("backtest", {})) if isinstance(outer.get("backtest"), Mapping) else {}
    supplied = {**outer, **nested}
    cfg = {**DEFAULT_BACKTEST_CONFIG, **supplied}
    aliases = {
        "annualization_days": "annualization",
        "execution_lag_days": "execution_lag",
        "signal_lag": "execution_lag",
    }
    for source, target in aliases.items():
        if source in supplied and target not in supplied:
            cfg[target] = supplied[source]
    if "drawdown_controls" in supplied and "drawdown" not in supplied:
        cfg["drawdown"] = {"drawdown_schedule": supplied["drawdown_controls"]}
    portfolio_cfg = outer.get("portfolio")
    if isinstance(portfolio_cfg, Mapping):
        cfg["portfolio"] = dict(portfolio_cfg)
        if "drawdown" not in cfg and "drawdown_controls" in portfolio_cfg:
            cfg["drawdown"] = {"drawdown_schedule": portfolio_cfg["drawdown_controls"]}
    return cfg


@dataclass
class BacktestResult:
    """Container returned by :func:`run_backtest`.

    It supports dictionary-style access for lightweight integration code while
    retaining typed attributes for notebooks and reporting.
    """

    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    target_weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.DataFrame
    drawdown: pd.Series
    metrics: dict[str, float]

    @property
    def weights(self) -> pd.DataFrame:
        """Executed portfolio weights (alias for ``positions``)."""

        return self.positions

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        return (
            "equity_curve",
            "returns",
            "gross_returns",
            "positions",
            "weights",
            "target_weights",
            "turnover",
            "costs",
            "drawdown",
            "metrics",
        )

    def items(self) -> Iterator[tuple[str, Any]]:
        for key in self.keys():
            yield key, getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.items())


def _datetime_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy()
    if "date" in result.columns and not isinstance(result.index, pd.DatetimeIndex):
        result = result.set_index("date")
    result.index = pd.to_datetime(result.index)
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def _coerce_returns(
    prices: pd.DataFrame | None,
    returns: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return simple asset returns without erasing gaps in price histories.

    A valid price following one or more missing marks is compared with the
    last valid price. The missing dates themselves remain missing, so the
    accounting loop can disclose any exposure that could not be priced. Log
    return inputs are converted to simple returns before portfolio P&L and
    wealth compounding.
    """

    if returns is None and prices is None:
        raise ValueError("provide prices or returns")
    method = str(config.get("return_method", "simple")).lower()
    if method not in {"simple", "pct", "percent", "log", "logarithmic"}:
        raise ValueError("return_method must be 'simple' or 'log'")

    if returns is not None:
        frame = _datetime_frame(returns, "returns").apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan)
        if method in {"log", "logarithmic"}:
            frame = np.expm1(frame)
    else:
        price_frame = _datetime_frame(prices, "prices").apply(pd.to_numeric, errors="coerce")
        # Invalid/non-positive marks cannot define a return. Carrying the last
        # valid mark only for the denominator preserves the cumulative move at
        # the next valid observation; the gap dates are restored to NaN below.
        valid_prices = price_frame.where(price_frame > 0.0)
        comparison_prices = valid_prices.ffill()
        if method in {"log", "logarithmic"}:
            frame = np.expm1(np.log(comparison_prices).diff())
        else:
            frame = comparison_prices.pct_change(fill_method=None)
        frame = frame.where(valid_prices.notna())
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _confidence_scale(frame: pd.DataFrame) -> pd.Series:
    if "confidence" not in frame.columns:
        return pd.Series(1.0, index=frame.index)
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    finite = confidence[np.isfinite(confidence)]
    if not finite.empty and finite.abs().quantile(0.95) > 1.5:
        confidence = confidence / 100.0
    return confidence.clip(0.0, 1.0)


def _find_return_column(asset: str, columns: pd.Index, config: Mapping[str, Any]) -> Any | None:
    if asset in columns:
        return asset
    alias_map = {**DEFAULT_ASSET_ALIASES, **dict(config.get("asset_aliases", {}))}
    candidates = alias_map.get(str(asset), (str(asset), str(asset).lower()))
    token_lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if str(candidate).lower() in token_lookup:
            return token_lookup[str(candidate).lower()]
    return None


def _returns_for_signal_assets(
    return_frame: pd.DataFrame,
    signals: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    requested = set(signals["asset"].dropna().astype(str)) if "asset" in signals else set()
    if "long_leg" in signals:
        requested.update(signals["long_leg"].dropna().astype(str))
    if "short_leg" in signals:
        requested.update(signals["short_leg"].dropna().astype(str))
    rename: dict[Any, str] = {}
    for asset in requested:
        column = _find_return_column(asset, return_frame.columns, config)
        if column is not None and column != asset:
            rename[column] = asset
    return return_frame.rename(columns=rename)


def _align_weight_columns(
    weights: pd.DataFrame,
    return_columns: pd.Index,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rename: dict[Any, Any] = {}
    for asset in weights.columns:
        column = _find_return_column(str(asset), return_columns, config)
        if column is not None:
            rename[asset] = column
    aligned = weights.rename(columns=rename)
    if aligned.columns.duplicated().any():
        aligned = aligned.T.groupby(level=0).sum().T
    return aligned


def _coerce_weights(
    return_frame: pd.DataFrame,
    target_weights: pd.DataFrame | None,
    signals: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if target_weights is not None:
        weights = target_weights.copy()
        if {"date", "asset", "weight"}.issubset(weights.columns):
            weights["date"] = pd.to_datetime(weights["date"])
            weights = weights.pivot_table(index="date", columns="asset", values="weight", aggfunc="mean")
        else:
            weights = _datetime_frame(weights, "target_weights")
    elif signals is not None:
        signal_frame = signals.copy()
        if {"date", "asset", "signal"}.issubset(signal_frame.columns):
            signal_frame["date"] = pd.to_datetime(signal_frame["date"])
            if bool(config.get("construct_portfolio", True)):
                portfolio_returns = _returns_for_signal_assets(return_frame, signal_frame, config)
                weights = construct_portfolio_history(
                    signal_frame,
                    portfolio_returns,
                    factor_exposures=config.get("factor_exposures"),
                    config=config.get("portfolio", {}),
                )
            else:
                scaled = pd.to_numeric(signal_frame["signal"], errors="coerce").fillna(0.0)
                scaled *= _confidence_scale(signal_frame)
                scratch = signal_frame.assign(_scaled=scaled)
                weights = scratch.pivot_table(index="date", columns="asset", values="_scaled", aggfunc="mean")
                gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
                weights = weights.div(gross, axis=0).fillna(0.0)
        else:
            weights = _datetime_frame(signal_frame, "signals").apply(pd.to_numeric, errors="coerce")
    else:
        raise ValueError("provide target_weights or signals")

    weights = weights.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    weights = _align_weight_columns(weights, return_frame.columns, config)
    common = weights.columns.intersection(return_frame.columns)
    if common.empty:
        raise ValueError("target weights and returns have no common assets")
    # NaN is a deliberate tri-state target: retain the prior instruction for
    # that asset. Explicit zero means flatten. This distinction prevents a
    # missing/untradable leg from becoming an artificial liquidation.
    return weights.reindex(columns=return_frame.columns).sort_index()


def _rebalance_targets(weights: pd.DataFrame, frequency: str) -> pd.DataFrame:
    frequency = str(frequency).lower()
    if frequency in {"daily", "d", "business_daily", "b"}:
        return weights
    rule = {"weekly": "W-FRI", "w": "W-FRI", "monthly": "ME", "m": "ME"}.get(frequency, frequency)
    def last_observed(group_rule: str) -> pd.DataFrame:
        selected: list[pd.DataFrame] = []
        for _, group in weights.groupby(pd.Grouper(freq=group_rule)):
            valid = group.dropna(how="all")
            if not valid.empty:
                # Preserve the actual decision timestamp. Resample(...).last()
                # relabels it at the period boundary and can add an unintended
                # execution day when month-end/week-end is not tradable.
                selected.append(valid.iloc[[-1]])
        return pd.concat(selected).sort_index() if selected else weights.iloc[0:0]

    try:
        return last_observed(rule)
    except ValueError:
        # pandas <2.2 uses M rather than ME.
        if rule == "ME":
            return last_observed("M")
        raise


def _align_rebalance_events(
    weights: pd.DataFrame,
    return_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map dated target decisions to the first tradable row on or after them.

    Unlike a forward-filled target matrix, this frame retains the distinction
    between a target *event* and a target merely remaining in force. That is
    necessary to avoid rebalancing drifted holdings on dates where no decision
    or scheduled rebalance occurred.
    """

    events = pd.DataFrame(np.nan, index=return_index, columns=weights.columns)
    if weights.empty or return_index.empty:
        return events

    for date, row in weights.sort_index().iterrows():
        location = int(return_index.searchsorted(date, side="left"))
        if location < len(return_index):
            # If several off-calendar decisions map to the same tradable row,
            # the latest decision wins.
            events.iloc[location] = row.to_numpy(dtype=float)
    return events


def _drift_weights(
    position: pd.Series,
    returns: pd.Series,
    portfolio_net_return: float,
) -> pd.Series:
    """Mark pre-return weights into post-return weights for the next decision.

    Trading costs reduce portfolio NAV and therefore enter the denominator,
    while instrument notionals are marked only by their asset returns. Missing
    returns are treated as unpriced (zero P&L for that period) and counted by
    :func:`run_backtest`.
    """

    growth = 1.0 + float(portfolio_net_return)
    if not np.isfinite(growth) or growth <= 0.0:
        # Portfolio weights are undefined after a complete gross loss.
        return pd.Series(0.0, index=position.index)
    marked = position * (1.0 + returns.fillna(0.0))
    return marked / growth


def _cost_vector(value: Any, assets: pd.Index, default: float) -> pd.Series:
    if isinstance(value, Mapping):
        result = pd.Series({str(asset): float(cost) for asset, cost in value.items()})
        return result.reindex(assets).fillna(float(default)) / 10_000.0
    scalar = default if value is None else float(value)
    return pd.Series(float(scalar) / 10_000.0, index=assets)


def _roll_matrix(index: pd.DatetimeIndex, assets: pd.Index, config: Mapping[str, Any]) -> pd.DataFrame:
    schedule = config.get("roll_schedule")
    result = pd.DataFrame(False, index=index, columns=assets)
    if schedule is None:
        return result
    if isinstance(schedule, pd.DataFrame):
        frame = schedule.copy()
        frame.index = pd.to_datetime(frame.index)
        return frame.reindex(index=index, columns=assets).fillna(False).astype(bool)
    if isinstance(schedule, Mapping):
        for asset, dates in schedule.items():
            if asset not in result.columns:
                continue
            parsed = pd.to_datetime(list(dates))
            result.loc[result.index.isin(parsed), asset] = True
        return result
    # A sequence of dates applies to configured futures, or all assets if no
    # explicit universe was supplied.
    dates = pd.to_datetime(list(schedule))
    futures = list(config.get("futures_assets", assets))
    columns = result.columns.intersection(futures)
    result.loc[result.index.isin(dates), columns] = True
    return result


def _annual_roll_vector(assets: pd.Index, config: Mapping[str, Any]) -> pd.Series:
    """Approximate continuous-futures rolls when exact dates are unavailable."""

    configured = config.get("annual_rolls", {})
    defaults = {
        "ZW": 4.0,
        "ZC": 4.0,
        "ZS": 4.0,
        "ZM": 4.0,
        "ZL": 4.0,
        "NG": 12.0,
        "CL": 12.0,
    }
    result = pd.Series(0.0, index=assets)
    for asset in assets:
        matched_code = None
        for code, aliases in DEFAULT_ASSET_ALIASES.items():
            if str(asset) == code or str(asset) in aliases:
                matched_code = code
                break
        if isinstance(configured, Mapping):
            value = configured.get(asset, configured.get(matched_code, defaults.get(matched_code, 0.0)))
        else:
            value = float(configured) if matched_code in defaults else 0.0
        result.loc[asset] = max(0.0, float(value))
    for asset in config.get("futures_assets", []):
        if asset in result.index and result.loc[asset] == 0.0:
            result.loc[asset] = 4.0
    return result


def _trade_statistics(positions: pd.DataFrame) -> tuple[int, float]:
    if positions.empty:
        return 0, 0.0
    total_trades = 0
    holding_periods: list[int] = []
    for asset in positions.columns:
        values = positions[asset].fillna(0.0).to_numpy()
        active_sign = np.sign(values)
        start: int | None = None
        prior = 0.0
        for i, sign in enumerate(active_sign):
            if sign != prior:
                if prior != 0 and start is not None:
                    holding_periods.append(i - start)
                if sign != 0:
                    total_trades += 1
                    start = i
                else:
                    start = None
                prior = sign
        if prior != 0 and start is not None:
            holding_periods.append(len(active_sign) - start)
    average_holding = float(np.mean(holding_periods)) if holding_periods else 0.0
    return total_trades, average_holding


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    positions: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Calculate investment metrics from net simple returns."""

    cfg = _backtest_config(config)
    series = pd.to_numeric(returns, errors="coerce").dropna()
    annualization = float(cfg["annualization"])
    if series.empty:
        keys = [
            "total_return", "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "maximum_drawdown", "calmar_ratio", "win_rate", "payoff_ratio",
            "annualized_turnover", "average_holding_period", "number_of_trades", "tail_loss",
            "var_95", "cvar_95",
        ]
        return {key: 0.0 for key in keys}

    wealth = (1.0 + series).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    years = len(series) / annualization
    if years > 0 and wealth.iloc[-1] > 0:
        annual_return = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        annual_return = np.nan
    annual_vol = float(series.std(ddof=1) * np.sqrt(annualization)) if len(series) > 1 else 0.0
    risk_free = float(cfg.get("risk_free_rate", 0.0))
    annual_excess_return = float(series.mean() * annualization - risk_free)
    sharpe = annual_excess_return / annual_vol if annual_vol > 0 else 0.0
    downside = series.where(series < 0.0, 0.0)
    downside_vol = float(np.sqrt((downside**2).mean()) * np.sqrt(annualization))
    sortino = annual_excess_return / downside_vol if downside_vol > 0 else 0.0
    drawdown = drawdown_series(series, "returns")
    maximum_drawdown = float(drawdown.min())
    calmar = annual_return / abs(maximum_drawdown) if maximum_drawdown < 0 and np.isfinite(annual_return) else 0.0
    active = series.loc[series != 0.0]
    win_rate = float((active > 0).mean()) if not active.empty else 0.0
    winners = active.loc[active > 0]
    losers = active.loc[active < 0]
    payoff = float(winners.mean() / abs(losers.mean())) if not winners.empty and not losers.empty else 0.0
    annual_turnover = float(turnover.fillna(0.0).mean() * annualization) if turnover is not None else 0.0
    number_trades, average_holding = _trade_statistics(positions) if positions is not None else (0, 0.0)
    confidence = float(np.clip(float(cfg.get("var_confidence", 0.95)), 0.50, 0.999))
    quantile = float(series.quantile(1.0 - confidence))
    tail = series.loc[series <= quantile]
    var = max(0.0, -quantile)
    cvar = max(0.0, -float(tail.mean())) if not tail.empty else var

    return {
        "total_return": total_return,
        "annualized_return": float(annual_return),
        "annualized_volatility": annual_vol,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": float(calmar),
        "win_rate": win_rate,
        "payoff_ratio": payoff,
        "annualized_turnover": annual_turnover,
        "average_holding_period": average_holding,
        "number_of_trades": float(number_trades),
        "tail_loss": max(0.0, -quantile),
        f"var_{int(confidence * 100)}": var,
        f"cvar_{int(confidence * 100)}": cvar,
    }


def run_backtest(
    prices: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
    target_weights: pd.DataFrame | None = None,
    config: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Run a next-period, cost-aware portfolio backtest.

    Signals/weights dated ``t`` are shifted by ``execution_lag`` before earning
    a return. A position is set at the open of an executable rebalance date,
    earns that row's return, and then drifts with its asset marks. On dates
    without a target event, the drifted holdings carry forward without an
    implicit rebalance; a changed drawdown-risk multiplier may still cause a
    trade. Drawdown controls are based on NAV through ``t-1``.

    Missing direct returns produce zero P&L only for the unpriced period and
    are disclosed in ``metrics``. For price inputs, the next valid mark earns
    the full move from the prior valid price. Trading, slippage, commission,
    and explicit futures-roll costs are reported separately.
    """

    cfg = _backtest_config(config)
    return_frame = _coerce_returns(prices, returns, cfg)
    if returns is None and prices is not None:
        price_marks = _datetime_frame(prices, "prices").apply(
            pd.to_numeric, errors="coerce"
        )
        mark_availability = price_marks.where(price_marks > 0.0).notna().reindex(
            index=return_frame.index,
            columns=return_frame.columns,
            fill_value=False,
        )
    else:
        # With direct returns, a missing value is the only available evidence
        # that an instrument could not be valued/traded for that interval.
        mark_availability = return_frame.notna()
    weights = _coerce_weights(return_frame, target_weights, signals, cfg)
    weights = _rebalance_targets(weights, str(cfg["rebalance_frequency"]))
    rebalance_events = _align_rebalance_events(weights, return_frame.index)
    target = rebalance_events.ffill().fillna(0.0)
    lag = max(0, int(cfg["execution_lag"]))
    executable_events = rebalance_events.shift(lag)
    event_mask = rebalance_events.notna().any(axis=1).shift(lag, fill_value=False)

    assets = return_frame.columns
    transaction_rate = _cost_vector(cfg.get("transaction_cost_bps"), assets, 2.0)
    slippage_rate = _cost_vector(cfg.get("slippage_bps"), assets, 1.0)
    commission_rate = _cost_vector(cfg.get("commission_bps"), assets, 0.0)
    roll_rate = _cost_vector(cfg.get("futures_roll_cost_bps"), assets, 1.0)
    roll_schedule = _roll_matrix(return_frame.index, assets, cfg)
    annual_rolls = _annual_roll_vector(assets, cfg)
    has_explicit_roll_schedule = cfg.get("roll_schedule") is not None

    dates = return_frame.index
    actual_positions = pd.DataFrame(0.0, index=dates, columns=assets)
    turnover_by_asset = pd.DataFrame(0.0, index=dates, columns=assets)
    costs = pd.DataFrame(
        0.0,
        index=dates,
        columns=["transaction_cost", "slippage", "commission", "roll_cost", "total_cost"],
    )
    gross_returns = pd.Series(0.0, index=dates, name="gross_return")
    net_returns = pd.Series(0.0, index=dates, name="return")
    equity = pd.Series(np.nan, index=dates, name="equity")

    capital = float(cfg["initial_capital"])
    peak = capital
    drifted_position = pd.Series(0.0, index=assets)
    active_target = pd.Series(0.0, index=assets)
    pending_execution = pd.Series(False, index=assets)
    active_multiplier: float | None = None
    missing_return_observations = 0
    unpriced_exposure_observations = 0
    unpriced_exposure_periods = 0
    drawdown_cfg = cfg.get("drawdown", cfg)
    for date in dates:
        raw_returns = return_frame.loc[date]
        priced = mark_availability.loc[date]
        pending_from_prior = pending_execution.copy()
        target_event = bool(event_mask.loc[date])
        instruction_mask = pd.Series(False, index=assets)
        if target_event:
            event_values = executable_events.loc[date].reindex(assets)
            supplied = event_values.notna()
            active_target.loc[supplied] = event_values.loc[supplied].astype(float)
            instruction_mask |= supplied
        prior_drawdown = capital / peak - 1.0 if peak > 0 else -1.0
        multiplier = (
            drawdown_risk_multiplier(prior_drawdown, drawdown_cfg)
            if bool(cfg.get("apply_drawdown_control", True))
            else 1.0
        )
        risk_multiplier_changed = (
            active_multiplier is not None
            and not np.isclose(multiplier, active_multiplier, rtol=0.0, atol=1e-12)
        )
        if risk_multiplier_changed:
            instruction_mask[:] = True
        desired_position = active_target * multiplier
        position = drifted_position.copy()
        if instruction_mask.any():
            differs = (desired_position - drifted_position).abs().gt(1e-12)
            # A previously deferred leg must first earn any catch-up move with
            # its old exposure. A fresh instruction on the repricing date may
            # update the desired target, but cannot move execution ahead of
            # that return.
            executable_before_return = (
                differs & priced & ~pending_from_prior & instruction_mask
            )
            position.loc[executable_before_return] = desired_position.loc[
                executable_before_return
            ]
            updated_pending = (differs & ~priced) | (pending_from_prior & priced)
            pending_execution.loc[instruction_mask] = updated_pending.loc[
                instruction_mask
            ]
        else:
            executable_before_return = pd.Series(False, index=assets)
        pre_return_trade = (position - drifted_position).abs()
        actual_positions.loc[date] = position

        if has_explicit_roll_schedule:
            roll_notional = position.abs().where(roll_schedule.loc[date], 0.0)
            costs.loc[date, "roll_cost"] = float(roll_notional.dot(roll_rate))
        elif bool(cfg.get("amortize_roll_cost", True)):
            daily_roll_rate = roll_rate * annual_rolls / float(cfg["annualization"])
            costs.loc[date, "roll_cost"] = float(position.abs().dot(daily_roll_rate))

        missing = raw_returns.isna()
        missing_return_observations += int(missing.sum())
        unpriced = ~priced & position.abs().gt(1e-12)
        unpriced_exposure_observations += int(unpriced.sum())
        unpriced_exposure_periods += int(unpriced.any())
        observed_returns = raw_returns.fillna(0.0)
        gross = float(position.dot(observed_returns))

        # A target deferred by a missing mark executes only after the first
        # subsequent valid mark. Existing exposure earns the catch-up return;
        # newly initiated notional cannot receive a move that happened before
        # it was tradable, and a pending reduction cannot evade that move.
        delayed_execution = pending_from_prior & priced
        estimated_pre_cost = float(
            pre_return_trade.dot(transaction_rate + slippage_rate + commission_rate)
            + costs.loc[date, "roll_cost"]
        )
        interim_net = max(-1.0, gross - estimated_pre_cost)
        marked_before_delayed_trade = _drift_weights(
            position,
            observed_returns,
            interim_net,
        )
        delayed_trade = pd.Series(0.0, index=assets)
        delayed_trade.loc[delayed_execution] = (
            desired_position.loc[delayed_execution]
            - marked_before_delayed_trade.loc[delayed_execution]
        ).abs()
        pending_execution.loc[delayed_execution] = False
        trade = pre_return_trade + delayed_trade
        turnover_by_asset.loc[date] = trade
        costs.loc[date, "transaction_cost"] = float(trade.dot(transaction_rate))
        costs.loc[date, "slippage"] = float(trade.dot(slippage_rate))
        costs.loc[date, "commission"] = float(trade.dot(commission_rate))
        costs.loc[date, "total_cost"] = float(costs.loc[date].iloc[:4].sum())

        net = gross - float(costs.loc[date, "total_cost"])
        # A fully lost NAV is terminal; clipping keeps the result well-defined
        # for pathological user inputs without fabricating a recovery.
        net = max(-1.0, net)
        gross_returns.loc[date] = gross
        net_returns.loc[date] = net
        capital *= 1.0 + net
        equity.loc[date] = capital
        peak = max(peak, capital)
        drifted_position = _drift_weights(position, observed_returns, net)
        drifted_position.loc[delayed_execution] = desired_position.loc[
            delayed_execution
        ]
        active_multiplier = float(multiplier)

    turnover = turnover_by_asset.sum(axis=1).rename("turnover")
    # Preserve the actual pre-return capital so the first realized loss is
    # measured against the starting NAV, including for non-unit capital.
    equity.attrs["initial_value"] = float(cfg["initial_capital"])
    drawdown = drawdown_series(equity, "equity")
    metrics = performance_metrics(net_returns, turnover, actual_positions, cfg)
    metrics["total_transaction_cost"] = float(costs["total_cost"].sum())
    metrics["average_gross_exposure"] = float(actual_positions.abs().sum(axis=1).mean())
    metrics["average_net_exposure"] = float(actual_positions.sum(axis=1).mean())
    metrics["missing_return_observations"] = float(missing_return_observations)
    metrics["unpriced_exposure_observations"] = float(unpriced_exposure_observations)
    metrics["unpriced_exposure_periods"] = float(unpriced_exposure_periods)

    return BacktestResult(
        equity_curve=equity,
        returns=net_returns,
        gross_returns=gross_returns,
        positions=actual_positions,
        target_weights=target,
        turnover=turnover,
        costs=costs,
        drawdown=drawdown,
        metrics=metrics,
    )


def slice_backtest_result(
    result: BacktestResult,
    start: Any,
    *,
    config: Mapping[str, Any] | None = None,
    asset_returns: pd.DataFrame | None = None,
) -> BacktestResult:
    """Rebase a backtest to an evaluation start while retaining warmup state.

    Targets and covariance estimates may be formed on a longer causal warmup
    history, but reported P&L, costs, drawdown, and metrics begin at ``start``.
    The retained first evaluation return is measured against the configured
    initial NAV rather than against the first post-return observation.
    """

    cfg = _backtest_config(config)
    cutoff = pd.Timestamp(start)
    index = result.returns.index[result.returns.index >= cutoff]
    if index.empty:
        raise ValueError("evaluation start is after the backtest history")

    returns = result.returns.reindex(index).copy()
    gross_returns = result.gross_returns.reindex(index).copy()
    positions = result.positions.reindex(index).copy()
    target_weights = result.target_weights.reindex(index).copy()
    turnover = result.turnover.reindex(index).copy()
    costs = result.costs.reindex(index).copy()
    initial = float(cfg.get("initial_capital", 1.0))
    equity = (initial * (1.0 + returns).cumprod()).rename("equity")
    equity.attrs["initial_value"] = initial
    drawdown = drawdown_series(equity, "equity")
    metrics = performance_metrics(returns, turnover, positions, cfg)
    metrics["total_transaction_cost"] = float(costs["total_cost"].sum())
    metrics["average_gross_exposure"] = float(positions.abs().sum(axis=1).mean())
    metrics["average_net_exposure"] = float(positions.sum(axis=1).mean())

    missing_observations = 0
    unpriced_observations = 0
    unpriced_periods = 0
    if asset_returns is not None:
        aligned_returns = _datetime_frame(asset_returns, "asset_returns").reindex(
            index=index,
            columns=positions.columns,
        )
        missing = aligned_returns.isna()
        unpriced = missing & positions.abs().gt(1e-12)
        missing_observations = int(missing.sum().sum())
        unpriced_observations = int(unpriced.sum().sum())
        unpriced_periods = int(unpriced.any(axis=1).sum())
    metrics["missing_return_observations"] = float(missing_observations)
    metrics["unpriced_exposure_observations"] = float(unpriced_observations)
    metrics["unpriced_exposure_periods"] = float(unpriced_periods)

    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        gross_returns=gross_returns,
        positions=positions,
        target_weights=target_weights,
        turnover=turnover,
        costs=costs,
        drawdown=drawdown,
        metrics=metrics,
    )


def walk_forward_splits(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create chronological train/test integer-index splits without overlap leakage."""

    n = len(index)
    train_size = int(train_size)
    test_size = int(test_size)
    step = int(step_size or test_size)
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, and step_size must be positive")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    test_start = train_size
    while test_start < n:
        train_start = 0 if expanding else max(0, test_start - train_size)
        test_end = min(n, test_start + test_size)
        train = np.arange(train_start, test_start, dtype=int)
        test = np.arange(test_start, test_end, dtype=int)
        if len(train) >= train_size and len(test) > 0:
            splits.append((train, test))
        test_start += step
    return splits


def _walk_forward_prediction(strategy: Any, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    model = copy.deepcopy(strategy)
    if hasattr(model, "fit") and hasattr(model, "predict"):
        model.fit(train)
        prediction = model.predict(test)
    elif callable(model):
        prediction = model(train, test)
    else:
        raise TypeError("strategy must be callable or implement fit/predict")
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()
    if not isinstance(prediction, pd.DataFrame):
        raise TypeError("walk-forward strategy must return a DataFrame or Series")
    prediction = prediction.copy()
    prediction.index = pd.to_datetime(prediction.index)
    return prediction.reindex(test.index)


def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | Any,
    config: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Fit/generate weights in chronological windows and backtest only OOS rows.

    A callable receives ``(train_prices, test_prices)`` and must return target
    weights indexed by the test dates.  Estimator-like objects may implement
    ``fit(train_prices)`` and ``predict(test_prices)`` instead.
    """

    cfg = dict(config or {})
    frame = _datetime_frame(prices, "prices").apply(pd.to_numeric, errors="coerce")
    train_size = int(cfg.get("train_size", 504))
    test_size = int(cfg.get("test_size", 63))
    splits = walk_forward_splits(
        frame.index,
        train_size=train_size,
        test_size=test_size,
        step_size=cfg.get("step_size"),
        expanding=bool(cfg.get("expanding", True)),
    )
    predictions: list[pd.DataFrame] = []
    for train_idx, test_idx in splits:
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]
        predictions.append(_walk_forward_prediction(strategy, train, test))
    if not predictions:
        raise ValueError("not enough observations for the requested walk-forward split")
    weights = pd.concat(predictions).loc[lambda x: ~x.index.duplicated(keep="first")].sort_index()
    oos_prices = frame.reindex(weights.index)
    backtest_cfg = dict(cfg.get("backtest", {}))
    # The prediction for a test date is still executed next period by default.
    return run_backtest(prices=oos_prices, target_weights=weights, config=backtest_cfg)


def compare_benchmarks(
    strategy: BacktestResult | pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: Mapping[str, pd.Series] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare a strategy with transparent static and momentum benchmarks."""

    cfg = dict(config or {})
    strategy_name = str(cfg.get("strategy_name", "strategy"))
    strategy_returns = strategy.returns if isinstance(strategy, BacktestResult) else pd.Series(strategy)
    assets = _datetime_frame(asset_returns, "asset_returns").apply(pd.to_numeric, errors="coerce")
    evaluation_start = (
        pd.Timestamp(cfg["evaluation_start"])
        if cfg.get("evaluation_start") is not None
        else None
    )
    # Benchmark accounting runs across the full supplied warm-up so drifted
    # holdings and the last scheduled rebalance survive the evaluation boundary.
    execution_assets = assets
    benchmark_targets: dict[str, pd.DataFrame] = {}
    if not assets.empty:
        equal_weight = pd.DataFrame(
            1.0 / len(assets.columns), index=assets.index, columns=assets.columns
        )
        benchmark_targets["equal_weight"] = equal_weight
    if not assets.empty:
        static_asset = str(cfg.get("static_long_asset", assets.columns[0]))
        if static_asset in assets.columns:
            static_target = pd.DataFrame(0.0, index=assets.index, columns=assets.columns)
            static_target[static_asset] = 1.0
            benchmark_targets[f"static_long_{static_asset}"] = static_target
        lookback = max(2, int(cfg.get("momentum_lookback", 60)))
        momentum_position = np.sign((1.0 + assets).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1.0)
        benchmark_targets["time_series_momentum"] = momentum_position / float(len(assets.columns))

    rows: dict[str, dict[str, float]] = {}
    if isinstance(strategy, BacktestResult):
        strategy_result = strategy
        if evaluation_start is not None and strategy.returns.index.min() < evaluation_start:
            strategy_result = slice_backtest_result(
                strategy,
                evaluation_start,
                config=cfg,
                asset_returns=assets,
            )
        rows[strategy_name] = dict(strategy_result.metrics)
    else:
        if evaluation_start is not None and isinstance(strategy_returns.index, pd.DatetimeIndex):
            strategy_returns = strategy_returns.loc[strategy_returns.index >= evaluation_start]
        rows[strategy_name] = performance_metrics(
            pd.to_numeric(strategy_returns, errors="coerce").dropna(), config=cfg
        )
    for name, target in benchmark_targets.items():
        result = run_backtest(
            returns=execution_assets,
            target_weights=target,
            config=cfg,
        )
        if evaluation_start is not None:
            result = slice_backtest_result(
                result,
                evaluation_start,
                config=cfg,
                asset_returns=assets,
            )
        rows[name] = dict(result.metrics)
    for name, values in dict(benchmark_returns or {}).items():
        # Externally supplied return-only benchmarks have no observable
        # position history, so turnover and costs remain unavailable/zero.
        external = pd.to_numeric(pd.Series(values), errors="coerce")
        if evaluation_start is not None and isinstance(external.index, pd.DatetimeIndex):
            external = external.loc[external.index >= evaluation_start]
        rows[str(name)] = performance_metrics(external.dropna(), config=cfg)
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "benchmark"
    return result.sort_values("sharpe_ratio", ascending=False)


# Familiar aliases.
backtest = run_backtest
calculate_metrics = performance_metrics
benchmark_comparison = compare_benchmarks


__all__ = [
    "BacktestResult",
    "DEFAULT_BACKTEST_CONFIG",
    "backtest",
    "benchmark_comparison",
    "calculate_metrics",
    "compare_benchmarks",
    "performance_metrics",
    "run_backtest",
    "slice_backtest_result",
    "walk_forward_backtest",
    "walk_forward_splits",
]

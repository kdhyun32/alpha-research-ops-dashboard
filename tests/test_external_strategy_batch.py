from __future__ import annotations

import pandas as pd

from scripts import external_strategy_batch as batch


def strategy(rule: str, *, symbol: str = "QQQ", required_symbols: list[str] | None = None) -> dict:
    return {
        "strategy_id": "test-strategy",
        "strategy_name": "test strategy",
        "traded_instrument": "TQQQ",
        "required_symbols": required_symbols or ["QQQ", "TQQQ"],
        "signals": [{"name": "signal", "symbol": symbol, "rule": rule, "role": "entry_filter"}],
        "entry_rule": "enter at next session open when signal is true",
        "exit_rule": "exit at next session open when signal is false",
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "benchmark": ["QQQ"],
    }


def test_multi_symbol_daily_rule_label_is_ready() -> None:
    normalized = batch.normalize_strategy(
        strategy(
            "QQQ close > QQQ 200-day simple moving average AND SPY close > SPY 200-day simple moving average",
            symbol="QQQ, SPY",
            required_symbols=["QQQ", "SPY", "TQQQ"],
        ),
        0,
    )

    assert normalized["validation_status"] == "ready_to_backtest"
    assert normalized["structured_rule_spec"]["type"] == "logical_and"
    assert "unsupported_signal_shape" not in normalized["parse_errors"]


def test_ratio_realized_volatility_and_count_rules_are_ready() -> None:
    cases = [
        strategy("QQQ>SMA200 and LQD/SHY ratio>SMA100", symbol="QQQ, LQD, SHY", required_symbols=["QQQ", "LQD", "SHY", "TQQQ"]),
        strategy("QQQ>SMA200 and QQQ RV21<35%", symbol="QQQ, RV", required_symbols=["QQQ", "RV", "TQQQ"]),
        strategy(
            "Risk-on if at least 2 of 4 are true: QQQ close > QQQ 200-day SMA; QQQ 126-day ROC > 0; VIX < 25; LQD 63-day ROC > SHY 63-day ROC",
            symbol="QQQ, VIX, LQD, SHY",
            required_symbols=["QQQ", "VIX", "LQD", "SHY", "TQQQ"],
        ),
    ]

    statuses = [batch.normalize_strategy(case, index)["validation_status"] for index, case in enumerate(cases)]

    assert statuses == ["ready_to_backtest", "ready_to_backtest", "ready_to_backtest"]


def test_bounded_cooldown_and_tiered_exposure_are_ready() -> None:
    cooldown = batch.normalize_strategy(
        strategy(
            "Use QQQ 200-day SMA trend gate for TQQQ exposure, but if QQQ 50-day z-score > 2.5 then hold zero TQQQ for a 5-session cooldown",
            required_symbols=["QQQ", "Z-SCORE", "TQQQ"],
        ),
        0,
    )
    tiered = batch.normalize_strategy(
        strategy(
            "Set TQQQ exposure to 100% if VIX close < 18, 50% if VIX close < 25, otherwise 0%",
            symbol="QQQ, VIX",
            required_symbols=["QQQ", "VIX", "TQQQ"],
        ),
        1,
    )

    assert cooldown["validation_status"] == "ready_to_backtest"
    assert cooldown["cooldown"]["trigger"]["type"] == "z_score"
    assert tiered["validation_status"] == "ready_to_backtest"
    assert tiered["exposure_rule"]["type"] == "tiered"


def test_formula_and_state_machine_rows_remain_held() -> None:
    target_vol = batch.normalize_strategy(
        strategy("QQQ>SMA200; TQQQ weight=min(1,25%/RV21)", symbol="QQQ, RV", required_symbols=["QQQ", "RV", "TQQQ"]),
        0,
    )
    reentry = batch.normalize_strategy(
        strategy(
            "Defensive if QQQ close < QQQ 100-day SMA AND QQQ 50-day SMA slope < 0; return to TQQQ after QQQ close > QQQ 200-day SMA",
            required_symbols=["QQQ", "TQQQ"],
        ),
        1,
    )

    assert target_vol["validation_status"] == "needs_formula_engine"
    assert reentry["validation_status"] == "needs_state_machine_support"


def test_ready_realized_volatility_rule_executes_with_synthetic_data(monkeypatch) -> None:
    normalized = batch.normalize_strategy(
        strategy("QQQ>SMA200 and QQQ RV21<35%", symbol="QQQ, RV", required_symbols=["QQQ", "RV", "TQQQ"]),
        0,
    )
    dates = pd.bdate_range("2020-01-01", periods=280)
    qqq = pd.DataFrame({"Open": range(100, 380), "Adj Close": range(100, 380), "Close": range(100, 380)}, index=dates)
    tqqq = pd.DataFrame({"Open": range(50, 330), "Adj Close": range(50, 330), "Close": range(50, 330)}, index=dates)

    def fake_download(symbols):
        return {"QQQ": qqq, "TQQQ": tqqq}, []

    monkeypatch.setattr(batch, "download_symbols", fake_download)

    result = batch.backtest(normalized)

    assert normalized["validation_status"] == "ready_to_backtest"
    assert result["status"] == "executed"
    assert result["metrics"]

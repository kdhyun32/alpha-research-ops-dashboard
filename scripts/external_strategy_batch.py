from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


BOUNDARY_FLAGS = {
    "data_source_tier": "exploratory_unofficial",
    "final_decision_allowed": False,
    "final_review_allowed": False,
    "licensed_rerun_required": True,
    "candidate_watch_champion_evidence": False,
    "ranking_applied": False,
    "recommendation_applied": False,
    "automatic_selection_applied": False,
    "candidate_watch_champion_mutation": False,
    "broker_live_order_alert_paper_automation": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("alpha-ext-%Y%m%d-%H%M%SZ")


def yfinance_version() -> dict[str, str]:
    try:
        return {"version": importlib.metadata.version("yfinance")}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"version_error": str(exc)}


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "strategy_batch" in payload:
        payload = payload["strategy_batch"]
    if "client_payload" in payload and "strategy_batch" in payload["client_payload"]:
        payload = payload["client_payload"]["strategy_batch"]
    if "strategies" not in payload:
        payload = {"schema_name": "external_strategy_ideas", "schema_version": "1.0", "strategies": [payload]}
    return payload


def as_number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default


def validate_strategy(strategy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("strategy_name", "traded_instrument", "required_symbols", "signals", "entry_rule", "exit_rule", "signal_timing", "execution_timing", "same_close_execution_allowed", "benchmark", "costs"):
        if key not in strategy or strategy[key] in (None, ""):
            errors.append(f"{key} 항목이 필요합니다.")
    if strategy.get("signal_timing") != "after_close":
        errors.append("signal_timing은 after_close여야 합니다.")
    if strategy.get("execution_timing") != "next_session_open":
        errors.append("execution_timing은 next_session_open이어야 합니다.")
    if strategy.get("same_close_execution_allowed") is not False:
        errors.append("same_close_execution_allowed는 false여야 합니다.")
    if not isinstance(strategy.get("required_symbols"), list) or not strategy.get("required_symbols"):
        errors.append("required_symbols는 1개 이상이어야 합니다.")
    if not isinstance(strategy.get("signals"), list) or not strategy.get("signals"):
        errors.append("signals는 1개 이상이어야 합니다.")
    for idx, signal in enumerate(strategy.get("signals") or []):
        for key in ("name", "symbol", "rule", "role"):
            if not str(signal.get(key, "")).strip():
                errors.append(f"signals[{idx}].{key} 항목이 필요합니다.")
        if not supported_rule(str(signal.get("rule", ""))):
            errors.append(f"signals[{idx}].rule은 현재 runner가 해석할 수 없는 규칙입니다.")
    costs = strategy.get("costs") or {}
    if not isinstance(costs.get("commission"), (int, float)):
        errors.append("costs.commission은 숫자여야 합니다.")
    if not isinstance(costs.get("slippage_per_trade"), (int, float)):
        errors.append("costs.slippage_per_trade는 숫자여야 합니다.")
    return errors


def supported_rule(rule: str) -> bool:
    lowered = rule.lower()
    return bool(re.search(r"close\s*>\s*.+?(\d+)[- ]?day.*(moving average|sma)", lowered) or re.search(r"(\d+)[- ]?day return is positive", lowered))


def download_symbols(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    version = yfinance_version()
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    access_timestamp = utc_now()
    for symbol in symbols:
        data = yf.download(symbol, start="2010-01-01", auto_adjust=False, progress=False, threads=False)
        if data.empty:
            raise RuntimeError(f"{symbol} 데이터를 yfinance에서 받지 못했습니다.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        basis = "adjusted" if "Adj Close" in data.columns else "unadjusted"
        if "Adj Close" not in data.columns:
            data["Adj Close"] = data["Close"]
        frames[symbol] = data.dropna(subset=["Open", "Adj Close"]).copy()
        audits.append({
            "symbol": symbol,
            "source_provider": "yfinance/Yahoo-family",
            "access_timestamp": access_timestamp,
            "yfinance_package": version,
            "row_count": int(len(frames[symbol])),
            "date_range": {"start": frames[symbol].index.min().strftime("%Y-%m-%d"), "end": frames[symbol].index.max().strftime("%Y-%m-%d")},
            "adjusted_unadjusted_basis": basis,
            "data_source_tier": "exploratory_unofficial",
        })
    return frames, audits


def signal_series(signal: dict[str, Any], frames: dict[str, pd.DataFrame]) -> pd.Series:
    symbol = signal["symbol"]
    rule = str(signal["rule"]).lower()
    close = frames[symbol]["Adj Close"]
    ma = re.search(r"close\s*>\s*.+?(\d+)[- ]?day.*(moving average|sma)", rule)
    if ma:
        days = int(ma.group(1))
        return close > close.rolling(days).mean()
    momentum = re.search(r"(\d+)[- ]?day return is positive", rule)
    if momentum:
        days = int(momentum.group(1))
        return close.pct_change(days) > 0
    raise RuntimeError(f"지원하지 않는 신호 규칙입니다: {signal.get('rule')}")


def max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0


def backtest(strategy: dict[str, Any], index: int) -> dict[str, Any]:
    symbols = sorted(set(strategy.get("required_symbols", []) + [strategy["traded_instrument"], strategy["benchmark"], "QQQ", "TQQQ"]))
    frames, source_audit = download_symbols(symbols)
    signal_parts = [signal_series(signal, frames) for signal in strategy["signals"]]
    signal = pd.concat(signal_parts, axis=1).all(axis=1).dropna()
    traded = frames[strategy["traded_instrument"]]
    qqq = frames["QQQ"]
    tqqq = frames["TQQQ"]
    df = pd.DataFrame({
        "signal": signal,
        "open": traded["Open"],
        "qqq_open": qqq["Open"],
        "tqqq_open": tqqq["Open"],
    }).dropna()
    df["exposure"] = df["signal"].astype(bool).shift(1, fill_value=False).astype(float)
    df["asset_return"] = df["open"].shift(-1) / df["open"] - 1.0
    df["qqq_return"] = df["qqq_open"].shift(-1) / df["qqq_open"] - 1.0
    df["tqqq_return"] = df["tqqq_open"].shift(-1) / df["tqqq_open"] - 1.0
    df = df.dropna()
    slippage = as_number((strategy.get("costs") or {}).get("slippage_per_trade"))
    commission = as_number((strategy.get("costs") or {}).get("commission"))
    trades = df["exposure"].diff().abs().fillna(df["exposure"].abs())
    df["strategy_return"] = df["asset_return"] * df["exposure"] - trades * (slippage + commission)
    equity = (1 + df["strategy_return"]).cumprod()
    qqq_equity = (1 + df["qqq_return"]).cumprod()
    tqqq_equity = (1 + df["tqqq_return"]).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    cagr_value = cagr(equity)
    mdd = max_drawdown(equity)
    return {
        "input_order": index + 1,
        "strategy_name": strategy["strategy_name"],
        "signal_summary": ", ".join(signal.get("name", "-") for signal in strategy["signals"]),
        "signal_count": len(strategy["signals"]),
        "period": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "trading_basis": "after_close signal / next_session_open execution",
        "status": "실행 완료",
        "metrics": {
            "total_return": total_return,
            "cagr": cagr_value,
            "max_drawdown": mdd,
            "mar": cagr_value / abs(mdd) if mdd else None,
            "trade_count": int(trades.sum()),
            "difference_vs_tqqq": total_return - float(tqqq_equity.iloc[-1] - 1),
            "difference_vs_qqq": total_return - float(qqq_equity.iloc[-1] - 1),
        },
        "signals": strategy["signals"],
        "entry_rule": strategy["entry_rule"],
        "exit_rule": strategy["exit_rule"],
        "exposure_rule": strategy.get("exposure_rule", "모든 신호가 참이면 노출 1, 아니면 0"),
        "required_symbols": strategy["required_symbols"],
        "cost_assumption": f"commission={commission}, slippage_per_trade={slippage}",
        "data_source_tier": "exploratory_unofficial",
        "limitations": ["탐색용 yfinance/Yahoo-family 결과입니다.", "최종 검토나 후보/watch/Champion 증거가 아닙니다."],
        "source_audit": source_audit,
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
    }


def write_outputs(payload: dict[str, Any], output_dir: Path, mode: str) -> dict[str, Any]:
    current_run_id = run_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    validations = []
    results = []
    for index, strategy in enumerate(payload["strategies"]):
        errors = validate_strategy(strategy)
        validations.append({"input_order": index + 1, "strategy_name": strategy.get("strategy_name", f"전략 {index + 1}"), "valid": not errors, "errors": errors})
        if mode == "backtest" and not errors:
            try:
                results.append(backtest(strategy, index))
            except Exception as exc:
                results.append({
                    "input_order": index + 1,
                    "strategy_name": strategy.get("strategy_name", f"전략 {index + 1}"),
                    "status": "실행 실패",
                    "errors": [str(exc)],
                    "data_source_tier": "exploratory_unofficial",
                    "signal_timing": "after_close",
                    "execution_timing": "next_session_open",
                    "same_close_execution_allowed": False,
                    "signals": strategy.get("signals", []),
                    "required_symbols": strategy.get("required_symbols", []),
                    "entry_rule": strategy.get("entry_rule", ""),
                    "exit_rule": strategy.get("exit_rule", ""),
                    "cost_assumption": json.dumps(strategy.get("costs", {}), ensure_ascii=False),
                })
    result_payload = {
        "schema_name": "external_strategy_results",
        "schema_version": "1.0",
        "run_id": current_run_id,
        "generated_at": utc_now(),
        "mode": mode,
        **BOUNDARY_FLAGS,
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "validations": validations,
        "results": results,
    }
    run_path = runs_dir / f"{current_run_id}.json"
    run_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path = output_dir / "index.json"
    if index_path.exists():
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_payload = {"schema_name": "external_strategy_results_index", "schema_version": "1.0", "runs": []}
    index_payload["runs"] = [{"run_id": current_run_id, "path": f"external_strategy_results/runs/{current_run_id}.json", "generated_at": result_payload["generated_at"], "mode": mode}] + index_payload.get("runs", [])[:24]
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"run_id": current_run_id, "run_path": str(run_path), "latest_path": str(output_dir / "latest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "backtest"], required=True)
    parser.add_argument("--payload-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("external_strategy_results"))
    args = parser.parse_args()
    payload = load_payload(args.payload_file)
    summary = write_outputs(payload, args.output_dir, args.mode)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

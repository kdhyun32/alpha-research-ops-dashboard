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


MAX_WORKER_BATCH_SIZE = 10
YFINANCE_ALIASES = {
    "VIX": "^VIX",
    "VXN": "^VXN",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VVIX": "^VVIX",
}
NEEDS_DATA_SOURCE_MAPPING = {"DGS10", "T10Y2Y", "DTB3", "NFCI", "HY_SPREAD"}
SUPPORTED_STATUS = "ready_to_backtest"
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
        payload = {
            "schema_name": "external_strategy_ideas",
            "schema_version": "1.0",
            "strategies": [payload],
        }
    return payload


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def as_number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else default


def unique_strings(values: list[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in output:
            output.append(text)
    return output


def normalize_symbol(symbol: Any) -> tuple[str, str | None]:
    raw = clean_text(symbol).upper()
    if not raw:
        return "", None
    if raw in NEEDS_DATA_SOURCE_MAPPING:
        return raw, "needs_data_source_mapping"
    return YFINANCE_ALIASES.get(raw, raw), None


def normalize_benchmarks(value: Any) -> tuple[list[str], str]:
    raw_values = value if isinstance(value, list) else [value]
    benchmarks = []
    for raw in raw_values:
        symbol, _status = normalize_symbol(raw)
        if symbol:
            benchmarks.append(symbol)
    benchmarks = unique_strings(benchmarks)
    primary = "QQQ" if "QQQ" in benchmarks else benchmarks[0] if benchmarks else ""
    return benchmarks, primary


def split_symbol_field(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[,;/]", clean_text(value)) if part.strip()]


def split_multi_symbol_signal(signal: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    symbols = split_symbol_field(signal.get("symbol"))
    if len(symbols) <= 1:
        return [dict(signal)], None
    rule = clean_text(signal.get("rule"))
    parts = re.split(r"\s+AND\s+", rule, flags=re.IGNORECASE)
    if len(parts) != len(symbols):
        return [dict(signal)], "unsupported_signal_shape"
    split_signals: list[dict[str, Any]] = []
    for idx, symbol in enumerate(symbols):
        part = parts[idx].strip()
        if not re.search(rf"\b{re.escape(symbol)}\b", part, flags=re.IGNORECASE):
            return [dict(signal)], "needs_rule_normalization"
        next_signal = dict(signal)
        next_signal["symbol"] = symbol
        next_signal["rule"] = part
        next_signal["name"] = clean_text(signal.get("name")) or f"{symbol} signal"
        split_signals.append(next_signal)
    return split_signals, None


def rule_support_status(rule: Any) -> str:
    lowered = clean_text(rule).lower()
    if not lowered:
        return "invalid_required_field"
    unsupported_tokens = (
        "percentile",
        "cooldown",
        "state machine",
        "score",
        "vote",
        "fraction",
        "rotation",
        "gld defensive",
        "fred",
        " or ",
        "window state",
    )
    if any(token in lowered for token in unsupported_tokens):
        return "unsupported_rule"
    if re.search(r"\b\d+[- ]?day\s+rate\s+of\s+change\s*>\s*0\b", lowered):
        return SUPPORTED_STATUS
    if re.search(r"\b\d+[- ]?day\s+return\s+is\s+positive\b", lowered):
        return SUPPORTED_STATUS
    if re.search(r"\bclose\s*>\s*.+?\b\d+[- ]?day\s+(simple\s+moving\s+average|moving\s+average|sma)\b", lowered):
        return SUPPORTED_STATUS
    return "unsupported_rule"


def skip_reason_for(status: str) -> str:
    return {
        "invalid_required_field": "필수 입력값이 부족합니다.",
        "needs_rule_normalization": "복합 규칙을 실행 가능한 단일 signal 규칙으로 정규화해야 합니다.",
        "unsupported_signal_shape": "여러 symbol 신호를 자동 분해할 수 없어 보류했습니다.",
        "needs_data_source_mapping": "yfinance-only runner에서 바로 실행 불가. 별도 데이터 소스 매핑 필요",
        "unsupported_rule": "현재 runner가 지원하지 않는 규칙입니다.",
        "batch_split_required": "10개 초과 실행은 dashboard에서 10개 이하 묶음으로 나눠야 합니다.",
    }.get(status, "")


def normalize_strategy(strategy: dict[str, Any], index: int, total_ready_hint: int = 0) -> dict[str, Any]:
    errors: list[str] = []
    source_symbols: list[str] = []
    normalized_symbols: list[str] = []
    if not isinstance(strategy, dict):
        return {
            "input_order": index + 1,
            "strategy_name": f"strategy {index + 1}",
            "validation_status": "invalid_required_field",
            "rule_support_status": "invalid_required_field",
            "skip_reason": skip_reason_for("invalid_required_field"),
            "source_symbols": [],
            "normalized_symbols": [],
            "normalized_strategy": {},
            "valid": False,
            "errors": ["strategy row must be an object."],
        }

    normalized = dict(strategy)
    benchmarks, primary_benchmark = normalize_benchmarks(strategy.get("benchmark"))
    normalized["benchmarks"] = benchmarks
    normalized["primary_benchmark"] = primary_benchmark
    normalized["benchmark"] = primary_benchmark

    for key in ("strategy_name", "traded_instrument", "entry_rule", "exit_rule", "signal_timing", "execution_timing"):
        if not clean_text(strategy.get(key)):
            errors.append(f"{key} is required.")
    if not benchmarks:
        errors.append("benchmark or benchmarks is required.")
    if strategy.get("signal_timing") != "after_close":
        errors.append("signal_timing must be after_close.")
    if strategy.get("execution_timing") != "next_session_open":
        errors.append("execution_timing must be next_session_open.")
    if strategy.get("same_close_execution_allowed") is not False:
        errors.append("same_close_execution_allowed must be false.")

    split_signals: list[dict[str, Any]] = []
    signal_shape_status: str | None = None
    raw_signals = strategy.get("signals")
    if not isinstance(raw_signals, list) or not raw_signals:
        errors.append("signals must contain at least one item.")
    else:
        for signal in raw_signals:
            if not isinstance(signal, dict):
                errors.append("each signal must be an object.")
                continue
            pieces, status = split_multi_symbol_signal(signal)
            if status:
                signal_shape_status = status
            split_signals.extend(pieces)

    normalized_signal_rows: list[dict[str, Any]] = []
    rule_statuses: list[str] = []
    data_mapping_needed = False
    for signal in split_signals:
        for key in ("name", "symbol", "rule", "role"):
            if not clean_text(signal.get(key)):
                errors.append(f"signal.{key} is required.")
        source_symbols.extend(split_symbol_field(signal.get("symbol")))
        symbol, symbol_status = normalize_symbol(signal.get("symbol"))
        if symbol_status == "needs_data_source_mapping":
            data_mapping_needed = True
        if symbol:
            normalized_symbols.append(symbol)
        rule_statuses.append(rule_support_status(signal.get("rule")))
        normalized_signal = dict(signal)
        normalized_signal["symbol"] = symbol or clean_text(signal.get("symbol"))
        normalized_signal_rows.append(normalized_signal)

    for value in strategy.get("required_symbols") or []:
        source_symbols.extend(split_symbol_field(value))
        symbol, symbol_status = normalize_symbol(value)
        if symbol_status == "needs_data_source_mapping":
            data_mapping_needed = True
        if symbol:
            normalized_symbols.append(symbol)

    traded_symbol, traded_status = normalize_symbol(strategy.get("traded_instrument"))
    if traded_status == "needs_data_source_mapping":
        data_mapping_needed = True
    if traded_symbol:
        normalized["traded_instrument"] = traded_symbol
        normalized_symbols.append(traded_symbol)

    for benchmark in benchmarks:
        normalized_symbols.append(benchmark)
        if benchmark in NEEDS_DATA_SOURCE_MAPPING:
            data_mapping_needed = True

    costs = strategy.get("costs") or {}
    if not isinstance(costs, dict):
        errors.append("costs must be an object.")
    else:
        if not isinstance(costs.get("commission"), (int, float)):
            errors.append("costs.commission must be numeric.")
        if not isinstance(costs.get("slippage_per_trade"), (int, float)):
            errors.append("costs.slippage_per_trade must be numeric.")

    normalized["signals"] = normalized_signal_rows
    normalized["required_symbols"] = unique_strings(normalized_symbols)
    source_symbols = unique_strings(source_symbols)
    normalized_symbols = unique_strings(normalized_symbols)

    if errors:
        status = "invalid_required_field"
    elif data_mapping_needed:
        status = "needs_data_source_mapping"
    elif signal_shape_status:
        status = signal_shape_status
    elif any(item == "unsupported_rule" for item in rule_statuses):
        status = "unsupported_rule"
    elif any(item == "needs_rule_normalization" for item in rule_statuses):
        status = "needs_rule_normalization"
    else:
        status = SUPPORTED_STATUS

    return {
        "input_order": index + 1,
        "strategy_name": clean_text(strategy.get("strategy_name")) or f"strategy {index + 1}",
        "validation_status": status,
        "rule_support_status": status if status != SUPPORTED_STATUS else SUPPORTED_STATUS,
        "skip_reason": "" if status == SUPPORTED_STATUS else skip_reason_for(status),
        "source_symbols": source_symbols,
        "normalized_symbols": normalized_symbols,
        "normalized_strategy": normalized,
        "valid": status == SUPPORTED_STATUS,
        "errors": errors,
        "batch_split_required": total_ready_hint > MAX_WORKER_BATCH_SIZE,
    }


def signal_series(signal: dict[str, Any], frames: dict[str, pd.DataFrame]) -> pd.Series:
    symbol = signal["symbol"]
    rule = clean_text(signal.get("rule")).lower()
    close = frames[symbol]["Adj Close"]
    ma = re.search(r"close\s*>\s*.+?(\d+)[- ]?day.*(simple\s+moving\s+average|moving average|sma)", rule)
    if ma:
        days = int(ma.group(1))
        return close > close.rolling(days).mean()
    momentum = re.search(r"(\d+)[- ]?day return is positive", rule)
    if momentum:
        days = int(momentum.group(1))
        return close.pct_change(days) > 0
    roc = re.search(r"(\d+)[- ]?day rate of change\s*>\s*0", rule)
    if roc:
        days = int(roc.group(1))
        return close.pct_change(days) > 0
    raise RuntimeError(f"Unsupported signal rule: {signal.get('rule')}")


def download_symbols(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    version = yfinance_version()
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    access_timestamp = utc_now()
    for symbol in symbols:
        data = yf.download(symbol, start="2010-01-01", auto_adjust=False, progress=False, threads=False)
        if data.empty:
            raise RuntimeError(f"No yfinance data returned for {symbol}.")
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
            "date_range": {
                "start": frames[symbol].index.min().strftime("%Y-%m-%d"),
                "end": frames[symbol].index.max().strftime("%Y-%m-%d"),
            },
            "adjusted_unadjusted_basis": basis,
            "data_source_tier": "exploratory_unofficial",
        })
    return frames, audits


def max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0


def backtest(normalized: dict[str, Any], index: int) -> dict[str, Any]:
    strategy = normalized["normalized_strategy"]
    symbols = unique_strings([
        *strategy.get("required_symbols", []),
        strategy["traded_instrument"],
        strategy["primary_benchmark"],
        "QQQ",
        "TQQQ",
    ])
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
        "input_order": normalized["input_order"],
        "strategy_name": strategy["strategy_name"],
        "validation_status": normalized["validation_status"],
        "normalized_strategy": strategy,
        "skip_reason": "",
        "source_symbols": normalized["source_symbols"],
        "normalized_symbols": normalized["normalized_symbols"],
        "rule_support_status": normalized["rule_support_status"],
        "signal_summary": ", ".join(signal.get("name", "-") for signal in strategy["signals"]),
        "signal_count": len(strategy["signals"]),
        "period": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "trading_basis": "after_close signal / next_session_open execution",
        "status": "executed",
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
        "exposure_rule": strategy.get("exposure_rule", "Exposure is 1 when all signals are true; otherwise 0."),
        "required_symbols": strategy["required_symbols"],
        "benchmarks": strategy["benchmarks"],
        "primary_benchmark": strategy["primary_benchmark"],
        "cost_assumption": f"commission={commission}, slippage_per_trade={slippage}",
        "data_source_tier": "exploratory_unofficial",
        "limitations": [
            "Exploratory yfinance/Yahoo-family result.",
            "Not final review evidence and not candidate/watch/Champion evidence.",
        ],
        "source_audit": source_audit,
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
    }


def skipped_result(normalized: dict[str, Any]) -> dict[str, Any]:
    strategy = normalized.get("normalized_strategy") or {}
    return {
        "input_order": normalized["input_order"],
        "strategy_name": normalized["strategy_name"],
        "validation_status": normalized["validation_status"],
        "normalized_strategy": strategy,
        "skip_reason": normalized["skip_reason"],
        "source_symbols": normalized["source_symbols"],
        "normalized_symbols": normalized["normalized_symbols"],
        "rule_support_status": normalized["rule_support_status"],
        "status": "skipped",
        "metrics": {},
        "signals": strategy.get("signals", []),
        "required_symbols": strategy.get("required_symbols", []),
        "benchmarks": strategy.get("benchmarks", []),
        "primary_benchmark": strategy.get("primary_benchmark", ""),
        "entry_rule": strategy.get("entry_rule", ""),
        "exit_rule": strategy.get("exit_rule", ""),
        "cost_assumption": json.dumps(strategy.get("costs", {}), ensure_ascii=False),
        "data_source_tier": "exploratory_unofficial",
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
    }


def chunk_ranges(ready_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for chunk_index, start in enumerate(range(0, len(ready_rows), MAX_WORKER_BATCH_SIZE), start=1):
        chunk = ready_rows[start : start + MAX_WORKER_BATCH_SIZE]
        chunks.append({
            "batch_index": chunk_index,
            "row_range": f"{chunk[0]['input_order']}-{chunk[-1]['input_order']}",
            "input_orders": [row["input_order"] for row in chunk],
            "strategy_count": len(chunk),
        })
    return chunks


def write_outputs(payload: dict[str, Any], output_dir: Path, mode: str) -> dict[str, Any]:
    current_run_id = run_id()
    raw_strategies = payload.get("strategies") or []
    first_pass = [normalize_strategy(strategy, index) for index, strategy in enumerate(raw_strategies)]
    ready_count = sum(1 for row in first_pass if row["valid"])
    validations = [
        normalize_strategy(strategy, index, total_ready_hint=ready_count)
        for index, strategy in enumerate(raw_strategies)
    ]
    ready_rows = [row for row in validations if row["valid"]]
    results = []
    for row in validations:
        if not row["valid"]:
            results.append(skipped_result(row))
            continue
        if mode != "backtest":
            continue
        try:
            results.append(backtest(row, row["input_order"] - 1))
        except Exception as exc:
            failed = skipped_result(row)
            failed["status"] = "execution_failed"
            failed["skip_reason"] = str(exc)
            results.append(failed)

    counts: dict[str, int] = {}
    for row in validations:
        counts[row["validation_status"]] = counts.get(row["validation_status"], 0) + 1
    batch_plan = chunk_ranges(ready_rows)
    result_payload = {
        "schema_name": "external_strategy_results",
        "schema_version": "1.1",
        "run_id": current_run_id,
        "generated_at": utc_now(),
        "mode": mode,
        **BOUNDARY_FLAGS,
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "validation_summary": {
            "total_rows": len(validations),
            "ready_to_backtest": ready_count,
            "skipped_or_held": len(validations) - ready_count,
            "status_counts": counts,
            "batch_size_limit": MAX_WORKER_BATCH_SIZE,
            "batch_split_required": ready_count > MAX_WORKER_BATCH_SIZE,
            "batch_count": len(batch_plan),
            "batch_plan": batch_plan,
        },
        "validations": validations,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{current_run_id}.json"
    run_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path = output_dir / "index.json"
    if index_path.exists():
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_payload = {"schema_name": "external_strategy_results_index", "schema_version": "1.0", "runs": []}
    index_payload["runs"] = [{
        "run_id": current_run_id,
        "path": f"external_strategy_results/runs/{current_run_id}.json",
        "generated_at": result_payload["generated_at"],
        "mode": mode,
        "validation_summary": result_payload["validation_summary"],
    }] + index_payload.get("runs", [])[:24]
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

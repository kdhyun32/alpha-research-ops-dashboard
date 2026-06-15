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


CANONICAL_SCHEMA_NAME = "alpha_research_strategy_batch"
CANONICAL_SCHEMA_VERSION = "2.0"
RESULT_SCHEMA_NAME = "alpha_research_strategy_batch_result"
RESULT_SCHEMA_VERSION = "2.0"
MAX_WORKER_BATCH_SIZE = 10
SUPPORTED_STATUS = "ready_to_backtest"

YFINANCE_ALIASES = {
    "VIX": "^VIX",
    "VXN": "^VXN",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VVIX": "^VVIX",
}
NEEDS_DATA_SOURCE_MAPPING = {"DGS10", "T10Y2Y", "DTB3", "NFCI", "HY_SPREAD", "FRED"}
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
    "ranking": False,
    "recommendation": False,
    "automatic_selection": False,
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


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    text = clean_text(value).replace("%", "")
    if not text:
        return default
    try:
        number = float(text)
    except ValueError:
        return default
    if "%" in clean_text(value) or abs(number) > 2:
        number /= 100.0
    return number if math.isfinite(number) else default


def unique_strings(values: list[Any]) -> list[str]:
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in output:
            output.append(text)
    return output


def split_symbol_field(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[,;/]", clean_text(value)) if part.strip()]


def normalize_symbol(symbol: Any) -> tuple[str, str | None]:
    raw = clean_text(symbol).upper()
    if not raw:
        return "", None
    if raw in NEEDS_DATA_SOURCE_MAPPING:
        return raw, "needs_data_source_mapping"
    return YFINANCE_ALIASES.get(raw, raw), None


def normalize_symbols(values: list[Any]) -> tuple[list[str], bool]:
    symbols: list[str] = []
    needs_mapping = False
    for value in values:
        for part in split_symbol_field(value):
            symbol, status = normalize_symbol(part)
            if status == "needs_data_source_mapping":
                needs_mapping = True
            if symbol:
                symbols.append(symbol)
    return unique_strings(symbols), needs_mapping


def normalize_benchmarks(strategy: dict[str, Any]) -> tuple[list[str], str, bool]:
    raw = strategy.get("benchmarks", strategy.get("benchmark", strategy.get("primary_benchmark")))
    values = raw if isinstance(raw, list) else [raw]
    benchmarks, needs_mapping = normalize_symbols(values)
    primary = clean_text(strategy.get("primary_benchmark"))
    primary, primary_status = normalize_symbol(primary)
    if primary_status == "needs_data_source_mapping":
        needs_mapping = True
    if not primary:
        primary = "QQQ" if "QQQ" in benchmarks else benchmarks[0] if benchmarks else ""
    if primary and primary not in benchmarks:
        benchmarks.insert(0, primary)
    return unique_strings(benchmarks), primary, needs_mapping


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "client_payload" in payload and "strategy_batch" in payload["client_payload"]:
        payload = payload["client_payload"]["strategy_batch"]
    if "strategy_batch" in payload:
        payload = payload["strategy_batch"]
    if "strategies" not in payload:
        payload = {"schema_name": CANONICAL_SCHEMA_NAME, "schema_version": CANONICAL_SCHEMA_VERSION, "strategies": [payload]}
    return payload


def operator_apply(left: pd.Series, operator: str, right: pd.Series | float) -> pd.Series:
    if operator in {">", "gt"}:
        return left > right
    if operator in {">=", "gte"}:
        return left >= right
    if operator in {"<", "lt"}:
        return left < right
    if operator in {"<=", "lte"}:
        return left <= right
    if operator in {"==", "=", "eq"}:
        return left == right
    raise RuntimeError(f"unsupported operator: {operator}")


def normalize_operator(value: Any) -> str:
    text = clean_text(value).lower()
    aliases = {
        "above": ">",
        "greater_than": ">",
        "below": "<",
        "less_than": "<",
        "at_or_below": "<=",
        "at_or_above": ">=",
    }
    return aliases.get(text, text or ">")


def make_node(node_type: str, **kwargs: Any) -> dict[str, Any]:
    return {"type": node_type, **{k: v for k, v in kwargs.items() if v is not None and v != ""}}


def strip_wrapping(text: str) -> str:
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def split_logical(text: str, keyword: str) -> list[str]:
    return [part.strip() for part in re.split(rf"\s+{keyword}\s+", text, flags=re.IGNORECASE) if part.strip()]


def parse_rule_text(rule: Any, default_symbol: Any = "") -> dict[str, Any] | None:
    text = strip_wrapping(clean_text(rule))
    if not text:
        return None
    lowered = text.lower()
    parts = split_logical(text, "or")
    if len(parts) > 1:
        children = [parse_rule_text(part, default_symbol) for part in parts]
        return make_node("logical_or", children=[child for child in children if child]) if all(children) else None
    parts = split_logical(text, "and")
    if len(parts) > 1:
        children = [parse_rule_text(part, default_symbol) for part in parts]
        return make_node("logical_and", children=[child for child in children if child]) if all(children) else None
    if lowered.startswith("not "):
        child = parse_rule_text(text[4:], default_symbol)
        return make_node("logical_not", child=child) if child else None

    default, _ = normalize_symbol(default_symbol)
    symbol = r"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)"
    op = r"(?P<op><=|>=|<|>|==|=)"

    day = r"[- ]?(?:trading[- ]?)?day"

    match = re.search(rf"{symbol}\s+(?:close\s+)?{op}\s+(?P=symbol)?\s*(?P<window>\d+){day}\s+(?:simple\s+moving\s+average|moving\s+average|sma)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("price_vs_sma", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(rf"(?:{symbol}\s+)?close\s+{op}\s+(?:sma|ma)\((?P<window>\d+)\)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol") or default)
        return make_node("price_vs_sma", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(rf"(?:sma|ma)\((?P<window>\d+)\)\s+slope\s+{op}\s+0", text, re.I)
    if match:
        return make_node("sma_slope", symbol=default, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+(?:rate of change|roc|total return|return)\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        threshold = as_number(match.group("threshold"))
        return make_node("roc_compare_zero" if threshold == 0 else "return_threshold", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=threshold)

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+(?:rate of change|roc|total return|return)\s+is\s+positive", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("roc_compare_zero", symbol=left, operator=">", window=int(match.group("window")))

    match = re.search(
        rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?:roc|return)\((?P<window>\d+)\)\s+{op}\s+(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?:roc|return)\((?P=window)\)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        return make_node("roc_compare_symbol", left_symbol=left, right_symbol=right, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(
        rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?P<window>\d+){day}\s+(?:rate of change|roc|total return|return)\s+{op}\s+(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?P=window){day}\s+(?:rate of change|roc|total return|return)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        return make_node("roc_compare_symbol", left_symbol=left, right_symbol=right, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(rf"{symbol}\s+close\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?)$", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("price_level_compare", symbol=left, operator=normalize_operator(match.group("op")), threshold=float(match.group("threshold")))

    match = re.search(
        rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+close\s+{op}\s+(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?P<window>\d+){day}\s+(?:simple\s+moving\s+average|moving\s+average|sma)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        sma_symbol, _ = normalize_symbol(match.group("right"))
        return make_node("price_vs_sma_other_symbol", symbol=left, sma_symbol=sma_symbol, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+annualized\s+realized\s+volatility\s+{op}\s+(?P<threshold>\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("realized_volatility", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+drawdown\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("trailing_drawdown", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+(?:total return|return)\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("return_threshold", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+percentile\s+{op}\s+(?P<threshold>\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("percentile", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(
        rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+close\s+{op}\s+trailing\s+(?P<window>\d+){day}\s+(?P<percentile>\d+(?:\.\d+)?)(?:th|st|nd|rd)?\s+percentile",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        percentile = float(match.group("percentile")) / 100.0
        return make_node("price_vs_trailing_percentile", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), percentile=percentile)

    match = re.search(
        rf"(?P<window>\d+){day}\s+(?:total return|return)\s+of\s+(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)/(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+ratio\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?%?)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        return make_node("ratio_return_threshold", left_symbol=left, right_symbol=right, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(rf"(?:last|recent)\s+(?P<window>\d+)\s+days?.*?(?:down|negative).*?(?:count\s*)?{op}\s*(?P<threshold>\d+)", text, re.I)
    if match:
        return make_node("count_condition", condition=make_node("down_day", symbol=default), window=int(match.group("window")), operator=normalize_operator(match.group("op")), threshold=int(match.group("threshold")))

    return None


def normalize_node(raw: Any, default_symbol: Any = "") -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if isinstance(raw, str):
        node = parse_rule_text(raw, default_symbol)
        if not node:
            return None, ["unsupported_rule"]
        return node, errors
    if not isinstance(raw, dict):
        return None, ["unsupported_rule"]
    if "rule" in raw and "type" not in raw:
        return normalize_node(raw.get("rule"), raw.get("symbol", default_symbol))
    node_type = clean_text(raw.get("type"))
    if not node_type:
        return None, ["unsupported_rule"]
    node = dict(raw)
    node["type"] = node_type
    for key in ("symbol", "left_symbol", "right_symbol", "sma_symbol"):
        if key in node:
            node[key], status = normalize_symbol(node[key])
            if status == "needs_data_source_mapping":
                errors.append("needs_data_source_mapping")
    if "operator" in node:
        node["operator"] = normalize_operator(node["operator"])
    if node_type in {"logical_and", "logical_or"}:
        children = []
        for child in raw.get("children") or []:
            parsed, child_errors = normalize_node(child, default_symbol)
            errors.extend(child_errors)
            if parsed:
                children.append(parsed)
        if not children:
            errors.append("unsupported_rule")
        node["children"] = children
    if node_type == "logical_not":
        parsed, child_errors = normalize_node(raw.get("child"), default_symbol)
        errors.extend(child_errors)
        if not parsed:
            errors.append("unsupported_rule")
        node["child"] = parsed
    if node_type == "count_condition":
        parsed, child_errors = normalize_node(raw.get("condition"), default_symbol)
        errors.extend(child_errors)
        if not parsed:
            errors.append("unsupported_rule")
        node["condition"] = parsed
    if node_type in {"vote", "votes", "hysteresis", "state_machine"}:
        errors.append("needs_state_machine_support")
    return node, errors


def node_symbols(node: Any) -> list[str]:
    if not isinstance(node, dict):
        return []
    symbols = []
    for key in ("symbol", "left_symbol", "right_symbol", "sma_symbol"):
        if clean_text(node.get(key)):
            symbols.append(clean_text(node[key]))
    for child in node.get("children") or []:
        symbols.extend(node_symbols(child))
    symbols.extend(node_symbols(node.get("child")))
    symbols.extend(node_symbols(node.get("condition")))
    return unique_strings(symbols)


def node_leaf_count(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    node_type = node.get("type")
    if node_type in {"logical_and", "logical_or"}:
        return sum(node_leaf_count(child) for child in node.get("children") or [])
    if node_type == "logical_not":
        return node_leaf_count(node.get("child"))
    if node_type == "count_condition":
        return node_leaf_count(node.get("condition"))
    return 1


def skip_reason_for(status: str) -> str:
    return {
        "invalid_required_field": "required field is missing or invalid",
        "needs_rule_normalization": "rule must be normalized into a supported signal graph",
        "unsupported_signal_shape": "multi-symbol signal shape could not be split safely",
        "needs_data_source_mapping": "requires a non-yfinance or separately mapped data source",
        "needs_state_machine_support": "stateful votes/hysteresis rule needs a state-machine implementation",
        "unsupported_rule": "rule family is not supported by this runner",
        "data_unavailable": "yfinance/Yahoo-family data was unavailable for at least one required symbol",
        "validation_only_not_executed": "validation mode only; no backtest was run",
    }.get(status, status)


def export_record_to_strategy(row: dict[str, Any]) -> dict[str, Any]:
    rule = row.get("rule_semantics") or {}
    signals = row.get("signals") or {}
    items = signals.get("signal_items") if isinstance(signals, dict) else row.get("signal_items")
    if not isinstance(items, list):
        items = []
    return {
        "strategy_id": row.get("strategy_id") or row.get("strategy_key"),
        "strategy_name": row.get("strategy_name") or row.get("strategy_id") or "imported strategy",
        "traded_instrument": rule.get("traded_instrument") or "TQQQ",
        "required_symbols": signals.get("required_inputs") if isinstance(signals, dict) else row.get("required_symbols", ["QQQ", "TQQQ"]),
        "signals": [
            {
                "name": item.get("name") or f"signal {index + 1}",
                "symbol": item.get("symbol") or "QQQ",
                "rule": item.get("rule") or item.get("rule_text") or item.get("description") or rule.get("source_rule_text"),
                "role": item.get("role") or "entry_filter",
            }
            for index, item in enumerate(items)
        ] or [
            {"name": row.get("strategy_name") or "imported signal", "symbol": "QQQ", "rule": rule.get("source_rule_text") or row.get("exposure_rule"), "role": "entry_filter"}
        ],
        "entry_rule": rule.get("entry_rule") or "enter at next session open when signal is true",
        "exit_rule": rule.get("exit_rule") or "exit at next session open when signal is false",
        "signal_timing": rule.get("signal_timing") or "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "benchmark": row.get("benchmark") or ["QQQ"],
        "costs": row.get("costs") or {"commission": 0, "slippage_per_trade": 0.0005},
        "exposure_rule": rule.get("exposure_rule") or row.get("exposure_rule"),
    }


def normalize_payload_strategies(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_name") == "alpha_research_strategy_selection_export":
        return [export_record_to_strategy(row) for row in payload.get("strategies") or []]
    rows = []
    for row in payload.get("strategies") or []:
        if isinstance(row, dict) and "strategy_idea" in row:
            rows.append(row["strategy_idea"])
        else:
            rows.append(row)
    return rows


def normalize_exposure(strategy: dict[str, Any], root_node: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    raw = strategy.get("exposure") or strategy.get("exposure_rule_spec")
    errors: list[str] = []
    if isinstance(raw, dict):
        exposure = dict(raw)
    else:
        exposure = {"type": "binary", "true_exposure": 1.0, "false_exposure": 0.0}
        text = clean_text(strategy.get("exposure_rule"))
        tier_matches = re.findall(r"(?:(\d+(?:\.\d+)?)%?\s+exposure).*?(?:when|if)\s+([^;]+)", text, flags=re.I)
        if tier_matches:
            tiers = []
            for exposure_text, condition_text in tier_matches:
                node, node_errors = normalize_node(condition_text, "")
                errors.extend(node_errors)
                if node:
                    value = float(exposure_text)
                    tiers.append({"when": node, "exposure": value / 100.0 if value > 1 else value})
            if tiers:
                exposure = {"type": "tiered", "tiers": tiers, "default_exposure": 0.0}
    if exposure.get("type") == "tiered":
        tiers = []
        for tier in exposure.get("tiers") or []:
            node, node_errors = normalize_node(tier.get("when") or tier.get("condition"), "")
            errors.extend(node_errors)
            if node:
                tiers.append({"when": node, "exposure": as_number(tier.get("exposure"))})
        exposure["tiers"] = tiers
        exposure["default_exposure"] = as_number(exposure.get("default_exposure"), 0.0)
    elif root_node:
        exposure["condition"] = root_node
        exposure["true_exposure"] = as_number(exposure.get("true_exposure"), 1.0)
        exposure["false_exposure"] = as_number(exposure.get("false_exposure"), 0.0)
    return exposure, errors


def normalize_cooldown(strategy: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    raw = strategy.get("cooldown") or strategy.get("lockout")
    if raw in (None, "", False):
        return None, []
    if not isinstance(raw, dict):
        return None, ["unsupported_rule"]
    node, errors = normalize_node(raw.get("trigger") or raw.get("condition"), "")
    if not node:
        errors.append("unsupported_rule")
    return {
        "trigger": node,
        "lockout_days": int(as_number(raw.get("lockout_days") or raw.get("days"), 0)),
        "exposure_during_lockout": as_number(raw.get("exposure_during_lockout"), 0.0),
    }, errors


def normalize_strategy(strategy: dict[str, Any], index: int, total_ready_hint: int = 0) -> dict[str, Any]:
    errors: list[str] = []
    parse_errors: list[str] = []
    source_symbols: list[str] = []
    if not isinstance(strategy, dict):
        errors.append("strategy row must be an object")
        strategy = {}

    for key in ("strategy_name", "traded_instrument", "signal_timing", "execution_timing"):
        if not clean_text(strategy.get(key)):
            errors.append(f"{key} is required")
    if strategy.get("signal_timing") != "after_close":
        errors.append("signal_timing must be after_close")
    if strategy.get("execution_timing") not in {"next_session_open", "next_session_open_to_next_session_open"}:
        errors.append("execution_timing must be next_session_open")
    if strategy.get("same_close_execution_allowed") is not False:
        errors.append("same_close_execution_allowed must be false")

    traded_symbol, traded_status = normalize_symbol(strategy.get("traded_instrument"))
    benchmarks, primary_benchmark, benchmark_needs_mapping = normalize_benchmarks(strategy)
    required_symbols, required_needs_mapping = normalize_symbols(strategy.get("required_symbols") or [])
    if not benchmarks:
        errors.append("benchmark or benchmarks is required")
    if not required_symbols:
        errors.append("required_symbols must contain at least one symbol")

    root_node: dict[str, Any] | None = None
    signal_rows = []
    raw_rule = strategy.get("rule_spec") or strategy.get("signal_graph")
    if raw_rule:
        root_node, node_errors = normalize_node(raw_rule, "")
        parse_errors.extend(node_errors)
    else:
        raw_signals = strategy.get("signals")
        if not isinstance(raw_signals, list) or not raw_signals:
            errors.append("signals must contain at least one item")
        else:
            child_nodes = []
            for signal_index, signal in enumerate(raw_signals, start=1):
                if not isinstance(signal, dict):
                    errors.append("each signal must be an object")
                    continue
                for key in ("name", "symbol", "rule", "role"):
                    if not clean_text(signal.get(key)):
                        errors.append(f"signal.{key} is required")
                source_symbols.extend(split_symbol_field(signal.get("symbol")))
                symbol, symbol_status = normalize_symbol(signal.get("symbol"))
                if symbol_status:
                    parse_errors.append(symbol_status)
                node, node_errors = normalize_node(signal.get("rule"), symbol)
                parse_errors.extend(node_errors)
                if node:
                    child_nodes.append(node)
                normalized_signal = dict(signal)
                normalized_signal["symbol"] = symbol or clean_text(signal.get("symbol"))
                normalized_signal["rule_spec"] = node
                signal_rows.append(normalized_signal)
            root_node = child_nodes[0] if len(child_nodes) == 1 else make_node("logical_and", children=child_nodes) if child_nodes else None

    exposure, exposure_errors = normalize_exposure(strategy, root_node)
    parse_errors.extend(exposure_errors)
    cooldown, cooldown_errors = normalize_cooldown(strategy)
    parse_errors.extend(cooldown_errors)

    normalized_symbols = unique_strings([
        *required_symbols,
        traded_symbol,
        *benchmarks,
        *node_symbols(root_node),
        *node_symbols(exposure),
        *node_symbols(cooldown),
        "QQQ",
        "TQQQ",
    ])
    source_symbols = unique_strings([*source_symbols, *strategy.get("required_symbols", []), strategy.get("traded_instrument"), *benchmarks])
    data_mapping_needed = (
        traded_status == "needs_data_source_mapping"
        or benchmark_needs_mapping
        or required_needs_mapping
        or "needs_data_source_mapping" in parse_errors
    )

    if errors:
        status = "invalid_required_field"
    elif data_mapping_needed:
        status = "needs_data_source_mapping"
    elif "needs_state_machine_support" in parse_errors:
        status = "needs_state_machine_support"
    elif parse_errors or not root_node:
        status = "unsupported_rule"
    else:
        status = SUPPORTED_STATUS

    normalized = {
        **strategy,
        "schema_name": CANONICAL_SCHEMA_NAME,
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "strategy_id": clean_text(strategy.get("strategy_id")) or f"import-{index + 1:03d}",
        "strategy_name": clean_text(strategy.get("strategy_name")) or f"strategy {index + 1}",
        "traded_instrument": traded_symbol or clean_text(strategy.get("traded_instrument")),
        "required_symbols": normalized_symbols,
        "benchmarks": benchmarks,
        "primary_benchmark": primary_benchmark,
        "benchmark": primary_benchmark,
        "signals": signal_rows or strategy.get("signals", []),
        "rule_spec": root_node,
        "signal_graph": root_node,
        "exposure": exposure,
        "cooldown": cooldown,
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "costs": strategy.get("costs") if isinstance(strategy.get("costs"), dict) else {"commission": 0, "slippage_per_trade": 0.0005},
        **BOUNDARY_FLAGS,
    }
    return {
        "input_order": index + 1,
        "strategy_id": normalized["strategy_id"],
        "strategy_name": normalized["strategy_name"],
        "validation_status": status,
        "rule_support_status": status,
        "skip_reason": "" if status == SUPPORTED_STATUS else skip_reason_for(status),
        "source_symbols": unique_strings(source_symbols),
        "normalized_symbols": normalized_symbols,
        "required_symbols": normalized_symbols,
        "structured_rule_spec": root_node,
        "exposure_rule": exposure,
        "cooldown": cooldown,
        "normalized_strategy": normalized,
        "valid": status == SUPPORTED_STATUS,
        "errors": errors,
        "parse_errors": unique_strings(parse_errors),
        "batch_split_required": total_ready_hint > MAX_WORKER_BATCH_SIZE,
    }


def download_symbols(symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    version = yfinance_version()
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    access_timestamp = utc_now()
    for symbol in unique_strings(symbols):
        data = yf.download(symbol, start="2010-01-01", auto_adjust=False, progress=False, threads=False)
        if data.empty:
            raise RuntimeError(f"data_unavailable: {symbol}")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        if "Adj Close" not in data.columns:
            data["Adj Close"] = data["Close"]
            basis = "unadjusted_close_fallback"
        else:
            basis = "adjusted"
        frame = data.dropna(subset=["Open", "Adj Close"]).copy()
        if frame.empty:
            raise RuntimeError(f"data_unavailable: {symbol}")
        frames[symbol] = frame
        audits.append({
            "symbol": symbol,
            "source_provider": "yfinance/Yahoo-family",
            "source_url_or_endpoint": f"yfinance.download({symbol})",
            "access_timestamp": access_timestamp,
            "retrieval_method": "yfinance.download",
            "local_file_path": None,
            "sha256": None,
            "row_count": int(len(frame)),
            "date_range": {"start": frame.index.min().strftime("%Y-%m-%d"), "end": frame.index.max().strftime("%Y-%m-%d")},
            "adjusted_unadjusted_basis": basis,
            "license_terms_or_redistribution_status": "Yahoo/yfinance exploratory access; terms not audited for redistribution",
            "yfinance_package": version,
            "data_source_tier": "exploratory_unofficial",
        })
    return frames, audits


def close_series(frames: dict[str, pd.DataFrame], symbol: str) -> pd.Series:
    if symbol not in frames:
        raise RuntimeError(f"missing frame for {symbol}")
    return frames[symbol]["Adj Close"]


def eval_node(node: dict[str, Any], frames: dict[str, pd.DataFrame]) -> pd.Series:
    node_type = node.get("type")
    if node_type == "logical_and":
        parts = [eval_node(child, frames) for child in node.get("children") or []]
        return pd.concat(parts, axis=1).all(axis=1)
    if node_type == "logical_or":
        parts = [eval_node(child, frames) for child in node.get("children") or []]
        return pd.concat(parts, axis=1).any(axis=1)
    if node_type == "logical_not":
        return ~eval_node(node["child"], frames).astype(bool)
    if node_type == "price_vs_sma":
        close = close_series(frames, node["symbol"])
        return operator_apply(close, node.get("operator", ">"), close.rolling(int(node["window"])).mean())
    if node_type == "sma_slope":
        close = close_series(frames, node["symbol"])
        sma = close.rolling(int(node["window"])).mean()
        return operator_apply(sma.diff(), node.get("operator", ">"), 0.0)
    if node_type == "roc_compare_zero":
        roc = close_series(frames, node["symbol"]).pct_change(int(node["window"]))
        return operator_apply(roc, node.get("operator", ">"), 0.0)
    if node_type == "roc_compare_symbol":
        left = close_series(frames, node["left_symbol"]).pct_change(int(node["window"]))
        right = close_series(frames, node["right_symbol"]).pct_change(int(node["window"]))
        return operator_apply(left, node.get("operator", ">"), right)
    if node_type == "price_level_compare":
        close = close_series(frames, node["symbol"])
        return operator_apply(close, node.get("operator", "<="), float(node["threshold"]))
    if node_type == "price_vs_sma_other_symbol":
        close = close_series(frames, node["symbol"])
        other = close_series(frames, node.get("sma_symbol") or node["symbol"]).rolling(int(node["window"])).mean()
        return operator_apply(close, node.get("operator", "<"), other)
    if node_type == "realized_volatility":
        close = close_series(frames, node["symbol"])
        vol = close.pct_change().rolling(int(node["window"])).std() * math.sqrt(float(node.get("annualization", 252)))
        return operator_apply(vol, node.get("operator", "<="), float(node["threshold"]))
    if node_type == "trailing_drawdown":
        close = close_series(frames, node["symbol"])
        drawdown = close / close.rolling(int(node["window"])).max() - 1.0
        return operator_apply(drawdown, node.get("operator", "<"), float(node["threshold"]))
    if node_type == "return_threshold":
        ret = close_series(frames, node["symbol"]).pct_change(int(node["window"]))
        return operator_apply(ret, node.get("operator", "<"), float(node["threshold"]))
    if node_type == "percentile":
        close = close_series(frames, node["symbol"])
        percentile = close.rolling(int(node["window"])).rank(pct=True)
        return operator_apply(percentile, node.get("operator", "<="), float(node["threshold"]))
    if node_type == "price_vs_trailing_percentile":
        close = close_series(frames, node["symbol"])
        threshold = close.rolling(int(node["window"])).quantile(float(node["percentile"]))
        return operator_apply(close, node.get("operator", "<="), threshold)
    if node_type == "ratio_return_threshold":
        ratio = close_series(frames, node["left_symbol"]) / close_series(frames, node["right_symbol"])
        ratio_return = ratio.pct_change(int(node["window"]))
        return operator_apply(ratio_return, node.get("operator", ">"), float(node["threshold"]))
    if node_type == "down_day":
        close = close_series(frames, node["symbol"])
        return close.pct_change() < 0
    if node_type == "count_condition":
        condition = eval_node(node["condition"], frames).astype(float)
        count = condition.rolling(int(node["window"])).sum()
        return operator_apply(count, node.get("operator", ">="), float(node["threshold"]))
    raise RuntimeError(f"Unsupported rule node: {node_type}")


def exposure_series(strategy: dict[str, Any], frames: dict[str, pd.DataFrame]) -> pd.Series:
    exposure = strategy.get("exposure") or {}
    root = strategy["rule_spec"]
    if exposure.get("type") == "tiered":
        base_index = eval_node(root, frames).index
        series = pd.Series(float(exposure.get("default_exposure", 0.0)), index=base_index)
        for tier in exposure.get("tiers") or []:
            mask = eval_node(tier["when"], frames).reindex(series.index).fillna(False).astype(bool)
            series.loc[mask] = float(tier.get("exposure", 0.0))
    else:
        condition = eval_node(exposure.get("condition") or root, frames).astype(bool)
        series = condition.astype(float) * float(exposure.get("true_exposure", 1.0))
        series = series.where(condition, float(exposure.get("false_exposure", 0.0)))
    cooldown = strategy.get("cooldown")
    if cooldown and cooldown.get("trigger") and int(cooldown.get("lockout_days") or 0) > 0:
        trigger = eval_node(cooldown["trigger"], frames).astype(bool).reindex(series.index, fill_value=False)
        lockout = pd.Series(False, index=series.index)
        days = int(cooldown["lockout_days"])
        for idx, active in enumerate(trigger.tolist()):
            if active:
                lockout.iloc[idx : idx + days] = True
        series.loc[lockout] = float(cooldown.get("exposure_during_lockout", 0.0))
    return series.clip(lower=0.0, upper=1.0)


def max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0


def skipped_result(normalized: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    strategy = normalized.get("normalized_strategy") or {}
    skip_reason = reason or normalized.get("skip_reason") or ""
    status = "skipped"
    return {
        "input_order": normalized["input_order"],
        "strategy_id": normalized.get("strategy_id"),
        "strategy_name": normalized.get("strategy_name"),
        "status": status,
        "skip_reason": skip_reason,
        "traded_instrument": strategy.get("traded_instrument", ""),
        "required_symbols": strategy.get("required_symbols", []),
        "benchmark": strategy.get("primary_benchmark", ""),
        "benchmarks": strategy.get("benchmarks", []),
        "primary_benchmark": strategy.get("primary_benchmark", ""),
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "period": "",
        "test_date_range": {"start": "", "end": ""},
        "signal_count": node_leaf_count(strategy.get("rule_spec")),
        "structured_rule_spec": strategy.get("rule_spec"),
        "exposure_rule": strategy.get("exposure"),
        "exposure_tiers_used": [],
        "metrics": {},
        "data_source_tier": "exploratory_unofficial",
        "final_review_allowed": False,
        "licensed_rerun_required": True,
        "ranking": False,
        "recommendation": False,
        "automatic_selection": False,
        "candidate_watch_champion_mutation": False,
        "normalized_strategy": strategy,
        "validation_status": normalized.get("validation_status"),
    }


def backtest(normalized: dict[str, Any]) -> dict[str, Any]:
    strategy = normalized["normalized_strategy"]
    symbols = unique_strings([*strategy.get("required_symbols", []), strategy["traded_instrument"], *strategy["benchmarks"], "QQQ", "TQQQ"])
    frames, source_audit = download_symbols(symbols)
    exposure_raw = exposure_series(strategy, frames)
    traded = frames[strategy["traded_instrument"]]
    df = pd.DataFrame({"raw_exposure": exposure_raw, "open": traded["Open"]}).dropna()
    df["exposure"] = df["raw_exposure"].shift(1, fill_value=0.0)
    df["asset_return"] = df["open"].shift(-1) / df["open"] - 1.0
    benchmark_returns: dict[str, pd.Series] = {}
    for benchmark in strategy["benchmarks"]:
        if benchmark in frames:
            aligned = frames[benchmark]["Open"].reindex(df.index)
            benchmark_returns[benchmark] = aligned.shift(-1) / aligned - 1.0
            df[f"benchmark_{benchmark}"] = benchmark_returns[benchmark]
    df = df.dropna(subset=["asset_return"])
    slippage = as_number((strategy.get("costs") or {}).get("slippage_per_trade"))
    commission = as_number((strategy.get("costs") or {}).get("commission"))
    trades = df["exposure"].diff().abs().fillna(df["exposure"].abs())
    df["strategy_return"] = df["asset_return"] * df["exposure"] - trades * (slippage + commission)
    equity = (1 + df["strategy_return"]).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    cagr_value = cagr(equity)
    mdd = max_drawdown(equity)
    benchmark_metrics: dict[str, dict[str, float]] = {}
    for benchmark in strategy["benchmarks"]:
        column = f"benchmark_{benchmark}"
        if column in df:
            bench_equity = (1 + df[column]).cumprod()
            benchmark_metrics[benchmark] = {
                "total_return": float(bench_equity.iloc[-1] - 1),
                "cagr": cagr(bench_equity),
                "max_drawdown": max_drawdown(bench_equity),
            }
    primary = strategy.get("primary_benchmark") or strategy["benchmarks"][0]
    primary_return = benchmark_metrics.get(primary, {}).get("total_return")
    qqq_return = benchmark_metrics.get("QQQ", {}).get("total_return")
    tqqq_return = benchmark_metrics.get("TQQQ", {}).get("total_return")
    tiers = sorted({float(value) for value in df["exposure"].dropna().unique().tolist()})
    return {
        "input_order": normalized["input_order"],
        "strategy_id": strategy["strategy_id"],
        "strategy_name": strategy["strategy_name"],
        "status": "executed",
        "skip_reason": "",
        "traded_instrument": strategy["traded_instrument"],
        "required_symbols": strategy["required_symbols"],
        "benchmark": primary,
        "benchmarks": strategy["benchmarks"],
        "primary_benchmark": primary,
        "signal_timing": "after_close",
        "execution_timing": "next_session_open",
        "same_close_execution_allowed": False,
        "period": f"{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}",
        "test_date_range": {"start": df.index.min().strftime("%Y-%m-%d"), "end": df.index.max().strftime("%Y-%m-%d")},
        "signal_count": node_leaf_count(strategy["rule_spec"]),
        "signal_summary": strategy.get("strategy_name"),
        "structured_rule_spec": strategy["rule_spec"],
        "signals": strategy.get("signals", []),
        "entry_rule": strategy.get("entry_rule", ""),
        "exit_rule": strategy.get("exit_rule", ""),
        "exposure_rule": strategy.get("exposure"),
        "exposure_tiers_used": tiers,
        "cooldown": strategy.get("cooldown"),
        "metrics": {
            "total_return": total_return,
            "cagr": cagr_value,
            "max_drawdown": mdd,
            "mdd": mdd,
            "mar": cagr_value / abs(mdd) if mdd else None,
            "trade_count": int(trades.sum()),
            "benchmark_returns": benchmark_metrics,
            "primary_benchmark_total_return": primary_return,
            "difference_vs_primary_benchmark": total_return - primary_return if primary_return is not None else None,
            "difference_vs_tqqq": total_return - tqqq_return if tqqq_return is not None else None,
            "difference_vs_qqq": total_return - qqq_return if qqq_return is not None else None,
            "difference_vs_tqqq_reference": total_return - tqqq_return if tqqq_return is not None else None,
            "difference_vs_qqq_reference": total_return - qqq_return if qqq_return is not None else None,
        },
        "data_source_tier": "exploratory_unofficial",
        "final_review_allowed": False,
        "licensed_rerun_required": True,
        "ranking": False,
        "recommendation": False,
        "automatic_selection": False,
        "candidate_watch_champion_mutation": False,
        "normalized_strategy": strategy,
        "source_audit": source_audit,
        "limitations": [
            "Exploratory yfinance/Yahoo-family result.",
            "Not final review evidence and not candidate/watch/Champion evidence.",
        ],
        **BOUNDARY_FLAGS,
    }


def chunk_ranges(ready_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for chunk_index, start in enumerate(range(0, len(ready_rows), MAX_WORKER_BATCH_SIZE), start=1):
        chunk = ready_rows[start : start + MAX_WORKER_BATCH_SIZE]
        chunks.append({"batch_index": chunk_index, "row_range": f"{chunk[0]['input_order']}-{chunk[-1]['input_order']}", "input_orders": [row["input_order"] for row in chunk], "strategy_count": len(chunk)})
    return chunks


def write_outputs(payload: dict[str, Any], output_dir: Path, mode: str) -> dict[str, Any]:
    current_run_id = run_id()
    raw_strategies = normalize_payload_strategies(payload)
    first_pass = [normalize_strategy(strategy, index) for index, strategy in enumerate(raw_strategies)]
    ready_count = sum(1 for row in first_pass if row["valid"])
    validations = [normalize_strategy(strategy, index, ready_count) for index, strategy in enumerate(raw_strategies)]
    ready_rows = [row for row in validations if row["valid"]]
    results = []
    for row in validations:
        if not row["valid"]:
            results.append(skipped_result(row))
        elif mode != "backtest":
            results.append(skipped_result(row, reason=skip_reason_for("validation_only_not_executed")))
        else:
            try:
                results.append(backtest(row))
            except Exception as exc:
                reason = str(exc)
                if "data_unavailable" in reason:
                    reason = reason
                results.append(skipped_result(row, reason=reason))

    counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for row in validations:
        counts[row["validation_status"]] = counts.get(row["validation_status"], 0) + 1
        if row["validation_status"] != SUPPORTED_STATUS:
            reason_counts[row["validation_status"]] = reason_counts.get(row["validation_status"], 0) + 1
    for row in results:
        if row["status"] == "skipped" and row.get("skip_reason", "").startswith("data_unavailable"):
            reason_counts["data_unavailable"] = reason_counts.get("data_unavailable", 0) + 1

    batch_plan = chunk_ranges(ready_rows)
    result_payload = {
        "schema_name": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "input_schema_name": CANONICAL_SCHEMA_NAME,
        "input_schema_version": CANONICAL_SCHEMA_VERSION,
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
            "hold_reason_counts": reason_counts,
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

from __future__ import annotations

import argparse
import hashlib
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
NEEDS_DATA_SOURCE_MAPPING = {
    "DGS10",
    "T10Y2Y",
    "DTB3",
    "NFCI",
    "HY_SPREAD",
    "FRED",
    "P066_AUTHORITY",
    "ABL_AUTHORITY",
}
DERIVED_PLACEHOLDER_SYMBOLS = {
    "RETURN",
    "VOLATILITY",
    "SLOPE",
    "MOMENTUM",
    "RATIO",
    "SMA50",
    "IF",
    "Z-SCORE",
    "ZSCORE",
}
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

PUBLIC_TEXT_REPLACEMENTS = {
    "CUSTOM_P25_A100_RAW": "external authority state map",
    "P25": "external parameter",
    "A100": "external parameter",
    "P066_AUTHORITY": "EXTERNAL_AUTHORITY",
    "P066": "external authority",
    "ABL_AUTHORITY": "EXTERNAL_AUTHORITY",
    "ABL": "external authority",
    "Snowball": "external prior system",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("alpha-ext-%Y%m%d-%H%M%SZ")


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sanitize_public_text(value: Any) -> Any:
    if isinstance(value, str):
        output = value
        for old, new in PUBLIC_TEXT_REPLACEMENTS.items():
            output = output.replace(old, new)
        return output
    if isinstance(value, list):
        return [sanitize_public_text(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_public_text(item) for key, item in value.items()}
    return value


def strategy_sequence_hash(strategies: list[dict[str, Any]]) -> str:
    sequence = [
        {
            "input_order": strategy.get("input_order", index + 1) if isinstance(strategy, dict) else index + 1,
            "strategy_id": clean_text(strategy.get("strategy_id") or strategy.get("strategy_key")) if isinstance(strategy, dict) else "",
            "strategy_name": clean_text(strategy.get("strategy_name")) if isinstance(strategy, dict) else "",
        }
        for index, strategy in enumerate(strategies)
    ]
    return stable_json_hash(sequence)


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
    if raw == "RV":
        return "", None
    if "," in raw:
        return raw, "unsupported_signal_shape"
    if raw in DERIVED_PLACEHOLDER_SYMBOLS:
        return raw, "unsupported_rule"
    if raw in NEEDS_DATA_SOURCE_MAPPING:
        return raw, "needs_data_source_mapping"
    return YFINANCE_ALIASES.get(raw, raw), None


def normalize_symbols(values: list[Any]) -> tuple[list[str], bool]:
    symbols: list[str] = []
    needs_mapping = False
    for value in values:
        for part in split_symbol_field(value):
            if part.strip().upper() in DERIVED_PLACEHOLDER_SYMBOLS or part.strip().upper() == "RV":
                continue
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


def qqq_trend(window: int = 200) -> dict[str, Any]:
    return make_node("price_vs_sma", symbol="QQQ", operator=">", window=window)


def qqq_momentum(window: int) -> dict[str, Any]:
    return make_node("roc_compare_zero", symbol="QQQ", operator=">", window=window)


def qqq_drawdown(operator: str, threshold: float) -> dict[str, Any]:
    return make_node("trailing_drawdown", symbol="QQQ", operator=operator, window=252, threshold=threshold)


def vix_level(operator: str, threshold: float, symbol: str = "^VIX") -> dict[str, Any]:
    return make_node("price_level_compare", symbol=symbol, operator=operator, threshold=threshold)


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
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=\w)(<=|>=|<|>)(?=\w|\d)", r" \1 ", text)
    ratio_sma_match = re.search(
        r"\b(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)/(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?:ratio\s*)?>\s*SMA(?P<window>\d+)\b",
        text,
        re.I,
    )
    if ratio_sma_match:
        left, _ = normalize_symbol(ratio_sma_match.group("left"))
        right, _ = normalize_symbol(ratio_sma_match.group("right"))
        return make_node("ratio_price_vs_sma", left_symbol=left, right_symbol=right, operator=">", window=int(ratio_sma_match.group("window")))
    text = re.sub(r"\b(\^?[A-Za-z][A-Za-z0-9.\-_]*)\s*>\s*SMA(\d+)\b", r"\1 close > \1 \2-day simple moving average", text, flags=re.I)
    text = re.sub(r"\b(\^?[A-Za-z][A-Za-z0-9.\-_]*)\s*<\s*SMA(\d+)\b", r"\1 close < \1 \2-day simple moving average", text, flags=re.I)
    text = re.sub(r"\bROC(\d+)\s*>\s*0\b", lambda m: f"{clean_text(default_symbol) or 'QQQ'} {m.group(1)}-day rate of change > 0", text, flags=re.I)
    text = re.sub(r"\b(\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+ROC(\d+)\b", r"\1 \2-day rate of change", text, flags=re.I)
    text = re.sub(r"\b(\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+RV(\d+)\s*<\s*(\d+(?:\.\d+)?%?)", r"\1 \2-day annualized realized volatility < \3", text, flags=re.I)
    text = re.sub(r"\bRV(\d+)\s*<\s*(\d+(?:\.\d+)?%?)", lambda m: f"{clean_text(default_symbol) or 'QQQ'} {m.group(1)}-day annualized realized volatility < {m.group(2)}", text, flags=re.I)
    lowered = text.lower()
    if any(token in lowered for token in ("p066_authority", "abl_authority", "authority engine", "health_score", "stress_score", "canonical grace")):
        return make_node("external_authority_state", symbol="P066_AUTHORITY")
    if lowered.startswith("arm after ") and "set a 63-session tqqq window" in lowered:
        return make_node(
            "armed_crash_reentry_timer",
            arm=make_node(
                "logical_or",
                children=[
                    make_node("price_level_compare", symbol="^VIX", operator=">", threshold=35.0),
                    make_node("trailing_drawdown", symbol="QQQ", operator="<", window=252, threshold=-0.18),
                ],
            ),
            recovery=make_node(
                "logical_and",
                children=[
                    make_node("price_change_threshold", symbol="^VIX", operator="<", window=10, threshold=0.0),
                    make_node("roc_compare_zero", symbol="QQQ", operator=">", window=10, threshold=0.0),
                    make_node("price_vs_sma", symbol="QQQ", operator=">", window=20),
                ],
            ),
            window_sessions=63,
            inside_half_condition=make_node("ratio_price_threshold", left_symbol="^VIX9D", right_symbol="^VIX3M", operator=">", threshold=1.15),
            inside_half_exposure=0.5,
            inside_full_exposure=1.0,
            outside_condition=make_node("price_vs_sma", symbol="QQQ", operator=">", window=100),
            outside_true_exposure=0.25,
            outside_false_exposure=0.0,
        )
    count_improved_match = re.search(
        r"count of (?P<symbols>[A-Z0-9./^,\s]+?) above (?P<sma>\d+)-day SMA has improved by at least (?P<threshold>\d+) versus (?P<lookback>\d+) sessions ago",
        text,
        re.I,
    )
    if count_improved_match:
        raw_symbols = re.split(r"[/,\s]+", count_improved_match.group("symbols").strip())
        symbols = []
        for raw_symbol in raw_symbols:
            symbol, status = normalize_symbol(raw_symbol)
            if symbol and not status:
                symbols.append(symbol)
        if symbols:
            count_node = make_node(
                "count_above_sma_improved",
                symbols=unique_strings(symbols),
                sma_window=int(count_improved_match.group("sma")),
                lookback=int(count_improved_match.group("lookback")),
                threshold=int(count_improved_match.group("threshold")),
            )
            if "qqq close > qqq 100-day sma" in lowered:
                return make_node("logical_and", children=[parse_rule_text("QQQ close > QQQ 100-day SMA", default_symbol), count_node])
            return count_node
    if any(token in lowered for token in ("state machine", "hysteresis", "maintain prior state", "when armed")):
        return make_node("state_machine")
    if "target annualized volatility" in lowered or "target-vol" in lowered or "target volatility" in lowered:
        return make_node("target_volatility_sizing")
    if "formula" in lowered or "compute " in lowered or "drag_score" in lowered:
        return make_node("formula_engine")
    if lowered.startswith("hold tqqq if ") and "otherwise select" in lowered:
        condition_text = re.split(r"\botherwise select\b", text[13:], flags=re.I, maxsplit=1)[0]
        return parse_rule_text(condition_text, default_symbol)
    if lowered.startswith("tqqq in trend") or lowered.startswith("tqqq in sma200 trend"):
        return parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol)
    if lowered.startswith("use qqq 200-day sma trend gate"):
        return parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol)
    if lowered.startswith("enter tqqq when "):
        return parse_rule_text(text[len("enter tqqq when "):].split(";")[0], default_symbol)
    if lowered.startswith("when ") and ", enter" in lowered:
        when_text = re.split(r",\s*enter", text[5:], flags=re.I, maxsplit=1)[0]
        enter_match = re.search(r"\bif\s+(.+?)(?:;|$)", text, re.I)
        children = [parse_rule_text(when_text, default_symbol)]
        if enter_match:
            children.append(parse_rule_text(enter_match.group(1), default_symbol))
        children = [child for child in children if child]
        return make_node("logical_and", children=children) if children else None
    if "full tqqq above sma200" in lowered:
        return parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol)
    if "qqq trend" in lowered and "vxn" in lowered:
        match = re.search(r"VXN\s*<\s*(\d+(?:\.\d+)?)", text, re.I)
        vxn_node = make_node("price_level_compare", symbol="^VXN", operator="<", threshold=float(match.group(1))) if match else None
        children = [parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol), vxn_node]
        return make_node("logical_and", children=[child for child in children if child])
    if "qqq trend" in lowered and "vvix percentile" in lowered:
        match = re.search(r"VVIX percentile\s*<\s*(\d+(?:\.\d+)?)", text, re.I)
        vvix_node = make_node("percentile", symbol="VVIX", operator="<", window=252, threshold=float(match.group(1)) / 100.0) if match else None
        return make_node("logical_and", children=[parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol), vvix_node])
    if "qqq momentum beats lqd/shy" in lowered:
        return make_node("logical_and", children=[
            parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol),
            parse_rule_text("QQQ 63-day rate of change > LQD 63-day rate of change", default_symbol),
            parse_rule_text("QQQ 63-day rate of change > SHY 63-day rate of change", default_symbol),
        ])
    if "xlk or soxx" in lowered and ("sma100" in lowered or "100-day" in lowered):
        return make_node("logical_and", children=[
            parse_rule_text("QQQ close > QQQ 200-day simple moving average", default_symbol),
            make_node("logical_or", children=[
                parse_rule_text("XLK close > XLK 100-day simple moving average", "XLK"),
                parse_rule_text("SOXX close > SOXX 100-day simple moving average", "SOXX"),
            ]),
        ])
    if "at the first daily signal check in each calendar month" in lowered:
        match = re.search(r"from\s+(.+?)\s+and hold", text, re.I)
        child = parse_rule_text(match.group(1), default_symbol) if match else None
        return make_node("monthly_revalidation", child=child) if child else None
    if lowered.startswith("if ") and "otherwise require" in lowered:
        require_text = re.split(r"\botherwise require\b", text, flags=re.I, maxsplit=1)[1]
        return parse_rule_text(require_text, default_symbol)
    if lowered.startswith("if ") and "otherwise allow" in lowered:
        return make_node("always_true", symbol=clean_text(default_symbol) or "QQQ")
    if "hy oas" in lowered or "hy_spread" in lowered:
        trend = parse_rule_text("QQQ close > QQQ 200-day simple moving average" if "sma200" in lowered else "QQQ close > QQQ 100-day simple moving average", default_symbol)
        return make_node("logical_and", children=[trend, make_node("price_level_compare", symbol="HY_SPREAD", operator="<", threshold=6.0)])
    if lowered.startswith("risk-on if at least"):
        match = re.search(r"at least\s+(?P<threshold>\d+)\s+of\s+(?P<total>\d+).*?true:\s*(?P<body>.+)", text, re.I)
        if match:
            children = [parse_rule_text(part, default_symbol) for part in re.split(r";", match.group("body")) if part.strip()]
            return make_node("count_condition", children=[child for child in children if child], operator=">=", threshold=int(match.group("threshold")))
    count_above_match = re.search(
        r"at least\s+(?P<threshold>\d+)\s+of\s+(?P<symbols>[A-Z0-9./^,\s]+?)\s+above\s+(?:their\s+)?SMA(?P<window>\d+)",
        text,
        re.I,
    )
    if count_above_match:
        raw_symbols = re.split(r"[/,\s]+", count_above_match.group("symbols").strip())
        children = []
        for raw_symbol in raw_symbols:
            symbol, status = normalize_symbol(raw_symbol)
            if symbol and not status:
                children.append(make_node("price_vs_sma", symbol=symbol, operator=">", window=int(count_above_match.group("window"))))
        return make_node("count_condition", children=children, operator=">=", threshold=int(count_above_match.group("threshold"))) if children else None
    if lowered.startswith("set tqqq exposure to the fraction"):
        return make_node("always_true", symbol=clean_text(default_symbol) or "QQQ")
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
    if not default and "," in clean_text(default_symbol):
        default_symbols, _ = normalize_symbols([default_symbol])
        default = default_symbols[0] if default_symbols else "QQQ"
    symbol = r"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)"
    op = r"(?P<op><=|>=|<|>|==|=)"

    day = r"[- ]?(?:trading[- ]?)?day"

    match = re.search(rf"{symbol}\s+(?:close\s+)?{op}\s+(?P=symbol)?\s*(?P<window>\d+){day}\s+(?:simple\s+moving\s+average|moving\s+average|sma)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("price_vs_sma", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")))

    match = re.search(rf"{symbol}\s+(?P<window>\d+){day}\s+annualized\s+realized\s+volatility\s+{op}\s+(?P<threshold>\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("realized_volatility", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(
        rf"{symbol}\s+close\s*-\s*(?P=symbol)\s+close\s+(?P<window>\d+)\s+sessions?\s+ago\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("price_change_threshold", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")))

    match = re.search(rf"{symbol}\s+one{day}\s+(?:rate of change|roc|total return|return)\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?%?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("return_threshold", symbol=left, operator=normalize_operator(match.group("op")), window=1, threshold=as_number(match.group("threshold")))

    match = re.search(rf"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?P<window>\d+)-day\s+z-score\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("z_score", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")))

    match = re.search(rf"{symbol}\s+close\s+percentile\s+over\s+the\s+trailing\s+(?P<window>\d+)(?:\s+trading)?\s+days\s+{op}\s+(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("percentile", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

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

    match = re.search(
        rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)/(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+ROC(?P<window>\d+)\s*{op}\s*(?P<threshold>-?\d+(?:\.\d+)?%?)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        return make_node("ratio_return_threshold", left_symbol=left, right_symbol=right, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=as_number(match.group("threshold")))

    match = re.search(rf"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)/SMA(?P<window>\d+)\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("distance_from_sma", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")) - 1.0)

    match = re.search(
        rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)/(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s*{op}\s*(?P<threshold>-?\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        return make_node("ratio_price_threshold", left_symbol=left, right_symbol=right, operator=normalize_operator(match.group("op")), threshold=float(match.group("threshold")))

    match = re.search(rf"{symbol}\s+close\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?)$", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("price_level_compare", symbol=left, operator=normalize_operator(match.group("op")), threshold=float(match.group("threshold")))

    match = re.search(rf"{symbol}\s+{op}\s+(?P<threshold>-?\d+(?:\.\d+)?)$", text, re.I)
    if match:
        if re.match(r"RSI\d+$", match.group("symbol"), re.I):
            window = int(re.search(r"\d+", match.group("symbol")).group(0))
            return make_node("rsi", symbol=clean_text(default_symbol) or "QQQ", operator=normalize_operator(match.group("op")), window=window, threshold=float(match.group("threshold")))
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

    match = re.search(rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)/(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?:ratio\s+)?{op}\s*(?P<threshold>-?\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        return make_node("ratio_price_threshold", left_symbol=left, right_symbol=right, operator=normalize_operator(match.group("op")), threshold=float(match.group("threshold")))

    match = re.search(rf"(?P<left>\^?[A-Za-z][A-Za-z0-9.\-_]*)/(?P<right>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?:ratio\s+)?(?:>|above)\s*(?:SMA|(?P=left)\s+(?P=right)\s+)?(?P<window>\d+)?", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("left"))
        right, _ = normalize_symbol(match.group("right"))
        window = int(match.group("window") or 100)
        return make_node("ratio_price_vs_sma", left_symbol=left, right_symbol=right, operator=">", window=window)

    match = re.search(rf"(?:fewer than|less than)\s+(?P<threshold>\d+)\s+of\s+the\s+last\s+(?P<window>\d+)\s+(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+daily returns are negative", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("count_condition", condition=make_node("down_day", symbol=left), window=int(match.group("window")), operator="<", threshold=int(match.group("threshold")))

    match = re.search(rf"(?:last|recent)\s+(?P<window>\d+)\s+days?.*?(?:down|negative).*?(?:count\s*)?{op}\s*(?P<threshold>\d+)", text, re.I)
    if match:
        return make_node("count_condition", condition=make_node("down_day", symbol=default), window=int(match.group("window")), operator=normalize_operator(match.group("op")), threshold=int(match.group("threshold")))

    match = re.search(rf"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+RSI(?P<window>\d+)\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("rsi", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")))

    match = re.search(rf"RSI(?P<window>\d+)\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        return make_node("rsi", symbol=clean_text(default_symbol) or "QQQ", operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")))

    match = re.search(rf"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)/SMA(?P<window>\d+)\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("distance_from_sma", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")) - 1.0)

    match = re.search(rf"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+(?P<window>\d+)-day\s+z-score\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("z_score", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")))

    match = re.search(rf"(?P<symbol>\^?[A-Za-z][A-Za-z0-9.\-_]*)\s+z(?P<window>\d+)\s*{op}\s*(?P<threshold>\d+(?:\.\d+)?)", text, re.I)
    if match:
        left, _ = normalize_symbol(match.group("symbol"))
        return make_node("z_score", symbol=left, operator=normalize_operator(match.group("op")), window=int(match.group("window")), threshold=float(match.group("threshold")))

    if "strongest positive 63-day momentum defensive asset" in lowered:
        return make_node("defensive_selector", assets=["SHY", "GLD", "LQD"], window=63)
    return None


def normalize_node(raw: Any, default_symbol: Any = "") -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if isinstance(raw, str):
        node = parse_rule_text(raw, default_symbol)
        if not node:
            return None, ["unsupported_rule"]
        if isinstance(node, dict):
            return normalize_node(node, default_symbol)
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
            elif status:
                errors.append(status)
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
        if raw.get("children"):
            children = []
            for child in raw.get("children") or []:
                parsed, child_errors = normalize_node(child, default_symbol)
                errors.extend(child_errors)
                if parsed:
                    children.append(parsed)
            if not children:
                errors.append("unsupported_rule")
            node["children"] = children
        else:
            parsed, child_errors = normalize_node(raw.get("condition"), default_symbol)
            errors.extend(child_errors)
            if not parsed:
                errors.append("unsupported_rule")
            node["condition"] = parsed
    if node_type == "armed_crash_reentry_timer":
        for key in ("arm", "recovery", "inside_half_condition", "outside_condition"):
            parsed, child_errors = normalize_node(raw.get(key), default_symbol)
            errors.extend(child_errors)
            if not parsed:
                errors.append("unsupported_rule")
            node[key] = parsed
        node["window_sessions"] = int(as_number(raw.get("window_sessions"), 0))
        node["inside_half_exposure"] = as_number(raw.get("inside_half_exposure"), 0.5)
        node["inside_full_exposure"] = as_number(raw.get("inside_full_exposure"), 1.0)
        node["outside_true_exposure"] = as_number(raw.get("outside_true_exposure"), 0.25)
        node["outside_false_exposure"] = as_number(raw.get("outside_false_exposure"), 0.0)
        if node["window_sessions"] <= 0:
            errors.append("unsupported_rule")
    if node_type == "count_above_sma_improved":
        symbols = []
        for raw_symbol in raw.get("symbols") or []:
            symbol, status = normalize_symbol(raw_symbol)
            if status == "needs_data_source_mapping":
                errors.append("needs_data_source_mapping")
            if symbol:
                symbols.append(symbol)
        node["symbols"] = unique_strings(symbols)
        node["sma_window"] = int(as_number(raw.get("sma_window"), 0))
        node["lookback"] = int(as_number(raw.get("lookback"), 0))
        node["threshold"] = int(as_number(raw.get("threshold"), 0))
        if not node["symbols"] or node["sma_window"] <= 0 or node["lookback"] <= 0 or node["threshold"] <= 0:
            errors.append("unsupported_rule")
    if node_type == "realized_volatility_compare":
        node["left_window"] = int(as_number(raw.get("left_window"), 0))
        node["right_window"] = int(as_number(raw.get("right_window"), 0))
        if node["left_window"] <= 0 or node["right_window"] <= 0:
            errors.append("unsupported_rule")
    if node_type == "source_rule_text":
        return normalize_node(raw.get("text"), default_symbol)
    if node_type in {"vote", "votes", "hysteresis", "state_machine", "needs_state_machine_support"}:
        errors.append("needs_state_machine_support")
    if node_type in {"target_volatility_sizing", "formula_engine"}:
        errors.append("needs_formula_engine")
    if node_type == "external_authority_state":
        errors.append("needs_data_source_mapping")
    if node_type == "price_level_compare" and clean_text(node.get("symbol")).upper() in DERIVED_PLACEHOLDER_SYMBOLS:
        errors.append("unsupported_rule")
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
    for key in ("child", "condition", "trigger", "arm", "recovery", "inside_half_condition", "outside_condition"):
        symbols.extend(node_symbols(node.get(key)))
    symbols.extend(clean_text(symbol) for symbol in node.get("symbols") or [])
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
        if node.get("children"):
            return sum(node_leaf_count(child) for child in node.get("children") or [])
        return node_leaf_count(node.get("condition"))
    return 1


def node_types(node: Any) -> set[str]:
    if not isinstance(node, dict):
        return set()
    output = {clean_text(node.get("type"))}
    for key in ("children",):
        for child in node.get(key) or []:
            output.update(node_types(child))
    for key in ("child", "condition", "trigger", "arm", "recovery", "inside_half_condition", "outside_condition"):
        output.update(node_types(node.get(key)))
    return {item for item in output if item}


def is_bounded_daily_node(node: Any) -> bool:
    allowed = {
        "always_true",
        "count_condition",
        "count_above_sma_improved",
        "distance_from_sma",
        "down_day",
        "logical_and",
        "logical_not",
        "logical_or",
        "monthly_revalidation",
        "percentile",
        "price_change_threshold",
        "price_level_compare",
        "price_vs_sma",
        "price_vs_sma_other_symbol",
        "price_vs_trailing_percentile",
        "ratio_price_threshold",
        "ratio_price_vs_sma",
        "ratio_return_threshold",
        "realized_volatility",
        "realized_volatility_compare",
        "return_threshold",
        "roc_compare_symbol",
        "roc_compare_zero",
        "rsi",
        "sma_slope",
        "trailing_drawdown",
        "variance_drag_score",
        "z_score",
    }
    symbols = {symbol.upper() for symbol in node_symbols(node)}
    return bool(node) and node_types(node).issubset(allowed) and not symbols.intersection(DERIVED_PLACEHOLDER_SYMBOLS)


def is_bounded_formula_pattern(strategy: dict[str, Any], combined_rule_text: str) -> bool:
    if any(token in combined_rule_text for token in ("target-vol", "target volatility", "weight=min", "return to tqqq after")):
        return False
    exposure = strategy.get("exposure") or {}
    allocation = strategy.get("allocation_rule") or {}
    root = strategy.get("rule_spec")
    if exposure.get("type") == "condition_fraction":
        conditions = exposure.get("conditions") or []
        return bool(conditions) and all(is_bounded_daily_node(condition) for condition in conditions)
    if exposure.get("type") == "tiered":
        tiers = exposure.get("tiers") or []
        return bool(tiers) and all(is_bounded_daily_node(tier.get("when")) for tier in tiers)
    if allocation and allocation.get("type") == "defensive_momentum_selector":
        assets = set(allocation.get("assets") or [])
        return assets.issubset({"SHY", "GLD", "LQD"}) and is_bounded_daily_node(allocation.get("condition") or root)
    return is_bounded_daily_node(exposure.get("condition") or root)


def is_bounded_state_pattern(strategy: dict[str, Any]) -> bool:
    exposure = strategy.get("exposure") or {}
    root = strategy.get("rule_spec")
    state_node = exposure if exposure.get("type") == "armed_crash_reentry_timer" else root
    if not isinstance(state_node, dict) or state_node.get("type") != "armed_crash_reentry_timer":
        return False
    return all(
        is_bounded_daily_node(state_node.get(key))
        for key in ("arm", "recovery", "inside_half_condition", "outside_condition")
    )


def skip_reason_for(status: str) -> str:
    return {
        "invalid_required_field": "required field is missing or invalid",
        "needs_rule_normalization": "rule must be normalized into a supported signal graph",
        "unsupported_signal_shape": "multi-symbol signal shape could not be split safely",
        "needs_data_source_mapping": "requires a non-yfinance or separately mapped data source",
        "needs_state_machine_support": "stateful votes/hysteresis rule needs a state-machine implementation",
        "needs_formula_engine": "formula-like sizing or target-volatility expression needs a bounded formula engine",
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
        text = " ".join(
            [clean_text(strategy.get("strategy_name")), clean_text(strategy.get("exposure_rule"))]
            + [clean_text(signal.get("rule")) for signal in strategy.get("signals") or [] if isinstance(signal, dict)]
        )
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
        if "set tqqq exposure to 100% if vix close < 18" in text.lower():
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": {"type": "price_level_compare", "symbol": "VIX", "operator": "<", "threshold": 18}, "exposure": 1.0},
                    {"when": {"type": "price_level_compare", "symbol": "VIX", "operator": "<", "threshold": 25}, "exposure": 0.5},
                ],
            }
        lowered_text = text.lower()
        if "vol term-structure state machine" in lowered_text and "use tqqq when recovery" in lowered_text:
            stress = make_node(
                "logical_or",
                children=[
                    make_node("ratio_price_threshold", left_symbol="^VIX9D", right_symbol="^VIX3M", operator=">", threshold=1.05),
                    make_node("ratio_price_threshold", left_symbol="^VIX", right_symbol="^VIX3M", operator=">", threshold=1.02),
                    vix_level(">", 115.0, "^VVIX"),
                    qqq_drawdown("<", -0.18),
                ],
            )
            recovery = make_node(
                "logical_and",
                children=[
                    make_node("ratio_price_threshold", left_symbol="^VIX9D", right_symbol="^VIX3M", operator="<", threshold=0.95),
                    make_node("price_change_threshold", symbol="^VIX", operator="<", window=10, threshold=0.0),
                    qqq_momentum(20),
                ],
            )
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": make_node("logical_not", child=stress), "exposure": 0.4},
                    {"when": make_node("logical_or", children=[recovery, qqq_trend(100)]), "exposure": 1.0},
                ],
            }
        if "damage recovery state machine" in lowered_text and "map panic-not-repaired" in lowered_text:
            panic = make_node(
                "logical_or",
                children=[
                    qqq_drawdown("<", -0.20),
                    make_node("trailing_drawdown", symbol="TQQQ", operator="<", window=252, threshold=-0.55),
                    vix_level(">", 35.0),
                ],
            )
            repaired = make_node(
                "logical_and",
                children=[
                    qqq_momentum(20),
                    qqq_trend(50),
                    make_node("price_change_threshold", symbol="^VIX", operator="<", window=10, threshold=0.0),
                ],
            )
            strong = make_node(
                "logical_and",
                children=[qqq_trend(100), qqq_drawdown(">", -0.10), vix_level("<", 28.0)],
            )
            exposure = {
                "type": "tiered",
                "default_exposure": 0.25,
                "tiers": [
                    {"when": make_node("logical_and", children=[panic, make_node("logical_not", child=repaired)]), "exposure": 0.0},
                    {"when": make_node("logical_and", children=[repaired, make_node("logical_not", child=strong)]), "exposure": 0.5},
                    {"when": strong, "exposure": 1.0},
                ],
            }
        if "breadth recovery ladder" in lowered_text and "positive 63-day momentum" in lowered_text:
            symbols = ["QQQ", "XLK", "SOXX", "SPY"]
            count63 = make_node(
                "count_condition",
                children=[make_node("roc_compare_zero", symbol=symbol, operator=">", window=63) for symbol in symbols],
                operator=">=",
                threshold=3,
            )
            count20 = make_node(
                "count_condition",
                children=[make_node("roc_compare_zero", symbol=symbol, operator=">", window=20) for symbol in symbols],
                operator=">=",
                threshold=3,
            )
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": count63, "exposure": 0.5},
                    {"when": make_node("logical_and", children=[count63, count20]), "exposure": 1.0},
                ],
            }
        if "variance drag allocator" in lowered_text and "compute drag_score" in lowered_text:
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": make_node("variance_drag_score", operator=">", threshold=0.02), "exposure": 0.5},
                    {"when": make_node("variance_drag_score", operator=">", threshold=0.10), "exposure": 1.0},
                ],
            }
        if "adaptive defensive rotation" in lowered_text and "risk count adds one each" in lowered_text:
            risk_count = [
                vix_level(">", 28.0),
                vix_level(">", 110.0, "^VVIX"),
                qqq_drawdown("<", -0.12),
                make_node("roc_compare_zero", symbol="QQQ", operator="<=", window=63),
            ]
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": make_node("count_condition", children=risk_count, operator="<=", threshold=1), "exposure": 1.0},
                    {"when": make_node("count_condition", children=risk_count, operator="==", threshold=2), "exposure": 0.35},
                ],
            }
        if "three-feature minimal regime" in lowered_text and "realized volatility 20-day" in lowered_text:
            trend_positive = make_node("logical_and", children=[qqq_trend(200), make_node("sma_slope", symbol="QQQ", operator=">", window=200)])
            rv20_le_rv60 = make_node("realized_volatility_compare", symbol="QQQ", left_window=20, right_window=60, operator="<=")
            rv20_gt_rv60 = make_node("realized_volatility_compare", symbol="QQQ", left_window=20, right_window=60, operator=">")
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": make_node("logical_and", children=[trend_positive, rv20_gt_rv60]), "exposure": 0.5},
                    {"when": make_node("logical_and", children=[trend_positive, rv20_le_rv60]), "exposure": 1.0},
                ],
            }
        if "half exposure when qqq/sma50" in text.lower():
            trend = parse_rule_text("QQQ close > QQQ 200-day simple moving average", "QQQ")
            overextended = parse_rule_text("QQQ/SMA50 > 1.12", "QQQ")
            exposure = {
                "type": "tiered",
                "default_exposure": 0.0,
                "tiers": [
                    {"when": trend, "exposure": 1.0},
                    {"when": make_node("logical_and", children=[trend, overextended]), "exposure": 0.5},
                ],
            }
        if "fraction of these true conditions" in text.lower():
            body = text.split(":", 1)[1] if ":" in text else text
            conditions = [normalize_node(part.strip(), "")[0] for part in re.split(r",", body) if part.strip()]
            conditions = [condition for condition in conditions if condition]
            if conditions:
                exposure = {"type": "condition_fraction", "conditions": conditions, "default_exposure": 0.0}
    if root_node and root_node.get("type") == "armed_crash_reentry_timer":
        exposure = dict(root_node)
    elif exposure.get("type") == "tiered":
        tiers = []
        for tier in exposure.get("tiers") or []:
            node, node_errors = normalize_node(tier.get("when") or tier.get("condition"), "")
            errors.extend(node_errors)
            if node:
                tiers.append({"when": node, "exposure": as_number(tier.get("exposure"))})
        exposure["tiers"] = tiers
        exposure["default_exposure"] = as_number(exposure.get("default_exposure"), 0.0)
        if not root_node and tiers:
            exposure["base_symbol"] = node_symbols({"children": [tier["when"] for tier in tiers]})[0]
    elif exposure.get("type") == "condition_fraction":
        conditions = []
        for condition in exposure.get("conditions") or []:
            node, node_errors = normalize_node(condition, "")
            errors.extend(node_errors)
            if node:
                conditions.append(node)
        exposure["conditions"] = conditions
        exposure["default_exposure"] = as_number(exposure.get("default_exposure"), 0.0)
    elif root_node:
        exposure["condition"] = root_node
        exposure["true_exposure"] = as_number(exposure.get("true_exposure"), 1.0)
        exposure["false_exposure"] = as_number(exposure.get("false_exposure"), 0.0)
    return exposure, errors


def normalize_allocation_rule(strategy: dict[str, Any], root_node: dict[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    raw = strategy.get("allocation_rule") or strategy.get("target_weights")
    errors: list[str] = []
    if isinstance(raw, dict):
        rule = dict(raw)
        for key in ("risk_on", "risk_off", "default"):
            weights = rule.get(key)
            if isinstance(weights, dict):
                rule[key] = {normalize_symbol(symbol)[0] or symbol: as_number(weight) for symbol, weight in weights.items()}
        return rule, errors
    if not root_node:
        return None, errors
    text = " ".join([clean_text(strategy.get("strategy_name")), clean_text(strategy.get("exposure_rule"))])
    for signal in strategy.get("signals") or []:
        if isinstance(signal, dict):
            text += " " + clean_text(signal.get("rule"))
    lowered = text.lower()
    defensive_asset = ""
    for asset in ("GLD", "SHY", "LQD", "BIL"):
        if re.search(rf"\belse\s+{asset.lower()}\b|/{asset.lower()}\b|->{asset.lower()}\b", lowered):
            defensive_asset = "SHY" if asset == "BIL" else asset
            break
    if "strongest positive 63-day momentum defensive asset" in lowered or "dynamic defensive selector" in lowered:
        return {
            "type": "defensive_momentum_selector",
            "condition": root_node,
            "risk_on": {"TQQQ": 1.0},
            "assets": ["SHY", "GLD", "LQD"],
            "window": 63,
            "cash_weight": 0.0,
        }, errors
    if defensive_asset:
        return {"type": "binary_target_weights", "condition": root_node, "risk_on": {"TQQQ": 1.0}, "risk_off": {defensive_asset: 1.0}}, errors
    if "else cash" in lowered or "otherwise 0%" in lowered or "otherwise 0" in lowered:
        return {"type": "binary_target_weights", "condition": root_node, "risk_on": {"TQQQ": 1.0}, "risk_off": {}}, errors
    return None, errors


def normalize_cooldown(strategy: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    raw = strategy.get("cooldown") or strategy.get("lockout")
    if raw in (None, "", False):
        text = " ".join([clean_text(strategy.get("exposure_rule"))] + [clean_text(signal.get("rule")) for signal in strategy.get("signals") or [] if isinstance(signal, dict)])
        lowered = text.lower()
        match = re.search(r"if\s+(.+?),?\s+set tqqq exposure to zero for\s+(\d+)\s+trading days", text, re.I)
        if not match:
            match = re.search(r"if\s+(.+?)\s+then hold zero tqqq for a\s+(\d+)-session cooldown", text, re.I)
        if match:
            node, errors = normalize_node(match.group(1), "")
            return {"trigger": node, "lockout_days": int(match.group(2)), "exposure_during_lockout": 0.0}, errors
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


def normalize_max_hold(strategy: dict[str, Any]) -> dict[str, Any] | None:
    text = " ".join([clean_text(strategy.get("exposure_rule"))] + [clean_text(signal.get("rule")) for signal in strategy.get("signals") or [] if isinstance(signal, dict)])
    match = re.search(r"position age (?:reaches|exceeds)\s+(\d+)\s+sessions", text, re.I)
    if not match:
        return None
    cooldown_match = re.search(r"then hold zero TQQQ for a\s+(\d+)-session cooldown", text, re.I)
    return {
        "type": "max_hold",
        "max_sessions": int(match.group(1)),
        "cooldown_sessions": int(cooldown_match.group(1)) if cooldown_match else 0,
    }


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
                symbol_values, symbol_needs_mapping = normalize_symbols([signal.get("symbol")])
                symbol = symbol_values[0] if symbol_values else clean_text(signal.get("symbol"))
                if symbol_needs_mapping:
                    parse_errors.append("needs_data_source_mapping")
                if clean_text(signal.get("symbol")) and "," in clean_text(signal.get("symbol")) and symbol_values:
                    symbol_status = None
                else:
                    symbol, symbol_status = normalize_symbol(signal.get("symbol"))
                if symbol_status and symbol_status != "unsupported_signal_shape":
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
    allocation_rule, allocation_errors = normalize_allocation_rule(strategy, root_node)
    parse_errors.extend(allocation_errors)
    cooldown, cooldown_errors = normalize_cooldown(strategy)
    parse_errors.extend(cooldown_errors)
    max_hold = normalize_max_hold(strategy)

    normalized_symbols = unique_strings([
        *required_symbols,
        traded_symbol,
        *benchmarks,
        *node_symbols(root_node),
        *node_symbols(exposure),
        *node_symbols(allocation_rule),
        *node_symbols(cooldown),
        "QQQ",
        "TQQQ",
    ])
    normalized_symbols = [symbol for symbol in normalized_symbols if symbol.upper() not in DERIVED_PLACEHOLDER_SYMBOLS]
    source_symbols = unique_strings([*source_symbols, *strategy.get("required_symbols", []), strategy.get("traded_instrument"), *benchmarks])
    mapping_symbols = {symbol.upper() for symbol in normalized_symbols if symbol.upper() in NEEDS_DATA_SOURCE_MAPPING}
    unsupported_placeholder_symbols = {
        symbol.upper()
        for symbol in normalized_symbols
        if symbol.upper() in DERIVED_PLACEHOLDER_SYMBOLS or "," in symbol
    }
    data_mapping_needed = (
        traded_status == "needs_data_source_mapping"
        or benchmark_needs_mapping
        or required_needs_mapping
        or "needs_data_source_mapping" in parse_errors
        or bool(mapping_symbols)
    )
    combined_rule_text = " ".join([clean_text(strategy.get("strategy_name")), clean_text(strategy.get("exposure_rule"))] + [clean_text(signal.get("rule")) for signal in strategy.get("signals") or [] if isinstance(signal, dict)]).lower()
    unsupported_runtime_derived = bool(unsupported_placeholder_symbols)
    formula_engine_needed = any(
        token in combined_rule_text
        for token in (
            "target-vol",
            "target volatility",
            "weight=min",
            "ladder",
            "fraction of these true conditions",
            "dynamic defensive",
            "composite",
            "risk count adds",
            "0% tqqq when",
        )
    ) or bool(re.search(r"\b(?:health|stress|trend[- ]volatility|volatility surface|trend)\s+score\b|score exposure", combined_rule_text))
    state_machine_needed = any(
        token in combined_rule_text
        for token in (
            "state machine",
            "stateful",
            "hysteresis",
            "maintain prior state",
            "when armed",
            "return to tqqq after",
            "re-entry",
            "improved by",
        )
    )
    tiered_exposure_ready = exposure.get("type") == "tiered" and bool(exposure.get("tiers"))
    effective_parse_errors = list(parse_errors)
    if tiered_exposure_ready and not root_node:
        effective_parse_errors = [error for error in effective_parse_errors if error != "unsupported_rule"]
    formula_context = {
        "rule_spec": root_node,
        "exposure": exposure,
        "allocation_rule": allocation_rule,
    }
    bounded_formula_ready = is_bounded_formula_pattern(formula_context, combined_rule_text)
    bounded_state_ready = is_bounded_state_pattern(formula_context)
    if bounded_formula_ready:
        effective_parse_errors = [
            error for error in effective_parse_errors if error != "needs_formula_engine"
        ]
    if bounded_state_ready:
        effective_parse_errors = [
            error for error in effective_parse_errors if error != "needs_state_machine_support"
        ]

    if errors:
        status = "invalid_required_field"
    elif data_mapping_needed:
        status = "needs_data_source_mapping"
    elif ("needs_state_machine_support" in effective_parse_errors or state_machine_needed) and not (bounded_state_ready or bounded_formula_ready):
        status = "needs_state_machine_support"
    elif ("needs_formula_engine" in effective_parse_errors or formula_engine_needed) and not bounded_formula_ready:
        status = "needs_formula_engine"
    elif unsupported_runtime_derived:
        status = "unsupported_rule"
    elif effective_parse_errors or (not root_node and not tiered_exposure_ready):
        status = "unsupported_rule"
    else:
        status = SUPPORTED_STATUS
    if status == "unsupported_rule" and formula_engine_needed:
        status = "needs_formula_engine"

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
        "allocation_rule": allocation_rule,
        "target_weights": allocation_rule,
        "cooldown": cooldown,
        "max_hold": max_hold,
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
        "max_hold": max_hold,
        "allocation_rule": allocation_rule,
        "target_weights": allocation_rule,
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
    if node_type == "price_change_threshold":
        close = close_series(frames, node["symbol"])
        change = close - close.shift(int(node["window"]))
        return operator_apply(change, node.get("operator", ">"), float(node["threshold"]))
    if node_type == "price_vs_sma_other_symbol":
        close = close_series(frames, node["symbol"])
        other = close_series(frames, node.get("sma_symbol") or node["symbol"]).rolling(int(node["window"])).mean()
        return operator_apply(close, node.get("operator", "<"), other)
    if node_type == "realized_volatility":
        close = close_series(frames, node["symbol"])
        vol = close.pct_change().rolling(int(node["window"])).std() * math.sqrt(float(node.get("annualization", 252)))
        return operator_apply(vol, node.get("operator", "<="), float(node["threshold"]))
    if node_type == "realized_volatility_compare":
        close = close_series(frames, node["symbol"])
        left = close.pct_change().rolling(int(node["left_window"])).std() * math.sqrt(float(node.get("annualization", 252)))
        right = close.pct_change().rolling(int(node["right_window"])).std() * math.sqrt(float(node.get("annualization", 252)))
        return operator_apply(left, node.get("operator", "<="), right)
    if node_type == "variance_drag_score":
        qqq = close_series(frames, "QQQ")
        vix9d = close_series(frames, "^VIX9D")
        vix3m = close_series(frames, "^VIX3M")
        momentum = qqq.pct_change(63)
        rv21 = qqq.pct_change().rolling(21).std() * math.sqrt(252)
        stress_penalty = (vix9d / vix3m > 1.05).astype(float) * 0.15
        score = momentum - 0.5 * (rv21 ** 2) - stress_penalty
        return operator_apply(score, node.get("operator", ">"), float(node["threshold"]))
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
    if node_type == "ratio_price_threshold":
        ratio = close_series(frames, node["left_symbol"]) / close_series(frames, node["right_symbol"])
        return operator_apply(ratio, node.get("operator", ">"), float(node["threshold"]))
    if node_type == "ratio_price_vs_sma":
        ratio = close_series(frames, node["left_symbol"]) / close_series(frames, node["right_symbol"])
        return operator_apply(ratio, node.get("operator", ">"), ratio.rolling(int(node["window"])).mean())
    if node_type == "rsi":
        close = close_series(frames, node["symbol"])
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(int(node["window"])).mean()
        loss = (-delta.clip(upper=0)).rolling(int(node["window"])).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, math.nan)))
        rsi = rsi.fillna(100)
        return operator_apply(rsi, node.get("operator", "<="), float(node["threshold"]))
    if node_type == "distance_from_sma":
        close = close_series(frames, node["symbol"])
        distance = close / close.rolling(int(node["window"])).mean() - 1.0
        return operator_apply(distance, node.get("operator", ">"), float(node["threshold"]))
    if node_type == "z_score":
        close = close_series(frames, node["symbol"])
        mean = close.rolling(int(node["window"])).mean()
        std = close.rolling(int(node["window"])).std()
        z = (close - mean) / std
        return operator_apply(z, node.get("operator", ">"), float(node["threshold"]))
    if node_type == "down_day":
        close = close_series(frames, node["symbol"])
        return close.pct_change() < 0
    if node_type == "count_condition":
        if node.get("children"):
            values = [eval_node(child, frames).astype(int) for child in node.get("children") or []]
            count = pd.concat(values, axis=1).sum(axis=1)
            return operator_apply(count, node.get("operator", ">="), float(node["threshold"]))
        condition = eval_node(node["condition"], frames).astype(float)
        count = condition.rolling(int(node.get("window", 1))).sum()
        return operator_apply(count, node.get("operator", ">="), float(node["threshold"]))
    if node_type == "count_above_sma_improved":
        values = []
        for symbol in node.get("symbols") or []:
            close = close_series(frames, symbol)
            values.append((close > close.rolling(int(node["sma_window"])).mean()).astype(int))
        count = pd.concat(values, axis=1).sum(axis=1)
        improvement = count - count.shift(int(node["lookback"]))
        return improvement >= int(node["threshold"])
    if node_type == "monthly_revalidation":
        child = eval_node(node["child"], frames).astype(bool)
        month_key = pd.Series(child.index.to_period("M"), index=child.index)
        first_signal = child.groupby(month_key).transform("first")
        return first_signal.astype(bool)
    if node_type == "always_true":
        return pd.Series(True, index=close_series(frames, node.get("symbol") or "QQQ").index)
    raise RuntimeError(f"Unsupported rule node: {node_type}")


def apply_max_hold(exposure: pd.Series, max_hold: dict[str, Any] | None) -> pd.Series:
    if not max_hold:
        return exposure
    max_sessions = int(max_hold.get("max_sessions") or 0)
    cooldown_sessions = int(max_hold.get("cooldown_sessions") or 0)
    if max_sessions <= 0:
        return exposure
    output = exposure.copy()
    active_age = 0
    cooldown_left = 0
    for idx, value in enumerate(exposure.tolist()):
        if cooldown_left > 0:
            output.iloc[idx] = 0.0
            cooldown_left -= 1
            active_age = 0
            continue
        if value > 0:
            active_age += 1
            if active_age > max_sessions:
                output.iloc[idx] = 0.0
                active_age = 0
                cooldown_left = cooldown_sessions
        else:
            active_age = 0
    return output


def exposure_series(strategy: dict[str, Any], frames: dict[str, pd.DataFrame]) -> pd.Series:
    exposure = strategy.get("exposure") or {}
    root = strategy["rule_spec"]
    if exposure.get("type") == "armed_crash_reentry_timer":
        base_index = eval_node(exposure["outside_condition"], frames).index
        arm = eval_node(exposure["arm"], frames).reindex(base_index).fillna(False).astype(bool)
        recovery = eval_node(exposure["recovery"], frames).reindex(base_index).fillna(False).astype(bool)
        half = eval_node(exposure["inside_half_condition"], frames).reindex(base_index).fillna(False).astype(bool)
        outside = eval_node(exposure["outside_condition"], frames).reindex(base_index).fillna(False).astype(bool)
        series = pd.Series(float(exposure.get("outside_false_exposure", 0.0)), index=base_index)
        armed = False
        window_left = 0
        window_sessions = int(exposure["window_sessions"])
        for idx, date in enumerate(base_index):
            if bool(arm.iloc[idx]):
                armed = True
            if armed and bool(recovery.iloc[idx]):
                window_left = window_sessions
                armed = False
            if window_left > 0:
                series.loc[date] = float(exposure.get("inside_half_exposure", 0.5)) if bool(half.iloc[idx]) else float(exposure.get("inside_full_exposure", 1.0))
                window_left -= 1
            else:
                series.loc[date] = float(exposure.get("outside_true_exposure", 0.25)) if bool(outside.iloc[idx]) else float(exposure.get("outside_false_exposure", 0.0))
    elif exposure.get("type") == "tiered":
        if root and is_bounded_daily_node(root):
            base_index = eval_node(root, frames).index
        else:
            base_symbol = exposure.get("base_symbol") or strategy.get("traded_instrument") or "QQQ"
            base_index = close_series(frames, base_symbol).index
        series = pd.Series(float(exposure.get("default_exposure", 0.0)), index=base_index)
        for tier in exposure.get("tiers") or []:
            mask = eval_node(tier["when"], frames).reindex(series.index).fillna(False).astype(bool)
            series.loc[mask] = float(tier.get("exposure", 0.0))
    elif exposure.get("type") == "condition_fraction":
        parts = [eval_node(condition, frames).astype(float) for condition in exposure.get("conditions") or []]
        if parts:
            series = pd.concat(parts, axis=1).mean(axis=1)
        else:
            series = eval_node(root, frames).astype(float)
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
    series = apply_max_hold(series, strategy.get("max_hold"))
    return series.clip(lower=0.0, upper=1.0)


def target_weight_frame(strategy: dict[str, Any], frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str], int]:
    allocation = strategy.get("allocation_rule")
    if not allocation:
        exposure = exposure_series(strategy, frames)
        return pd.DataFrame({strategy["traded_instrument"]: exposure}, index=exposure.index), [strategy["traded_instrument"]], 0
    condition = eval_node(allocation.get("condition") or strategy["rule_spec"], frames).astype(bool)
    condition = condition.reindex(condition.index).fillna(False)
    assets = unique_strings([
        *list((allocation.get("risk_on") or {}).keys()),
        *list((allocation.get("risk_off") or {}).keys()),
        *(allocation.get("assets") or []),
    ])
    weights = pd.DataFrame(0.0, index=condition.index, columns=assets)
    if allocation.get("type") == "defensive_momentum_selector":
        risk_on = allocation.get("risk_on") or {"TQQQ": 1.0}
        for symbol, weight in risk_on.items():
            weights.loc[condition, symbol] = float(weight)
        defensive_assets = allocation.get("assets") or ["SHY", "GLD", "LQD"]
        window = int(allocation.get("window") or 63)
        momentum = pd.DataFrame({symbol: close_series(frames, symbol).pct_change(window).reindex(condition.index) for symbol in defensive_assets})
        for date, row in momentum.loc[~condition].iterrows():
            positive = row[row > 0].dropna()
            if not positive.empty:
                weights.loc[date, positive.idxmax()] = 1.0
    else:
        for symbol, weight in (allocation.get("risk_on") or {"TQQQ": 1.0}).items():
            weights.loc[condition, symbol] = float(weight)
        for symbol, weight in (allocation.get("risk_off") or {}).items():
            weights.loc[~condition, symbol] = float(weight)
    if strategy.get("cooldown"):
        base = exposure_series({**strategy, "allocation_rule": None}, frames)
        weights = weights.mul(base.reindex(weights.index).fillna(0.0), axis=0)
    if strategy.get("max_hold"):
        gate = apply_max_hold(pd.Series(1.0, index=weights.index).where(weights.sum(axis=1) > 0, 0.0), strategy.get("max_hold"))
        weights = weights.mul(gate, axis=0)
    changes = int(weights.diff().abs().fillna(weights.abs()).sum(axis=1).gt(0).sum())
    return weights, assets, changes


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
    allocation = strategy.get("allocation_rule") or {}
    symbols = unique_strings([
        *strategy.get("required_symbols", []),
        strategy["traded_instrument"],
        *strategy["benchmarks"],
        *list((allocation.get("risk_on") or {}).keys()),
        *list((allocation.get("risk_off") or {}).keys()),
        *(allocation.get("assets") or []),
        "QQQ",
        "TQQQ",
    ])
    frames, source_audit = download_symbols(symbols)
    raw_weights, held_instruments, allocation_change_count = target_weight_frame(strategy, frames)
    shifted_weights = raw_weights.shift(1, fill_value=0.0)
    returns = pd.DataFrame(index=shifted_weights.index)
    for symbol in shifted_weights.columns:
        aligned = frames[symbol]["Open"].reindex(shifted_weights.index)
        returns[symbol] = aligned.shift(-1) / aligned - 1.0
    df = pd.DataFrame(index=shifted_weights.index)
    for symbol in shifted_weights.columns:
        df[f"weight_{symbol}"] = shifted_weights[symbol]
        df[f"return_{symbol}"] = returns[symbol]
    benchmark_returns: dict[str, pd.Series] = {}
    for benchmark in strategy["benchmarks"]:
        if benchmark in frames:
            aligned = frames[benchmark]["Open"].reindex(df.index)
            benchmark_returns[benchmark] = aligned.shift(-1) / aligned - 1.0
            df[f"benchmark_{benchmark}"] = benchmark_returns[benchmark]
    return_columns = [f"return_{symbol}" for symbol in shifted_weights.columns]
    df = df.dropna(subset=return_columns, how="all")
    slippage = as_number((strategy.get("costs") or {}).get("slippage_per_trade"))
    commission = as_number((strategy.get("costs") or {}).get("commission"))
    weight_columns = [f"weight_{symbol}" for symbol in shifted_weights.columns]
    weight_frame = df[weight_columns].copy()
    weight_frame.columns = shifted_weights.columns
    return_frame = df[return_columns].copy()
    return_frame.columns = shifted_weights.columns
    trades = weight_frame.diff().abs().fillna(weight_frame.abs()).sum(axis=1)
    df["strategy_return"] = (return_frame.fillna(0.0) * weight_frame).sum(axis=1) - trades * (slippage + commission)
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
    exposure_totals = weight_frame.sum(axis=1)
    tiers = sorted({float(value) for value in exposure_totals.dropna().unique().tolist()})
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
        "allocation_rule": strategy.get("allocation_rule"),
        "target_weights": strategy.get("target_weights"),
        "held_instruments": held_instruments,
        "allocation_change_count": allocation_change_count,
        "cooldown": strategy.get("cooldown"),
        "max_hold": strategy.get("max_hold"),
        "metrics": {
            "total_return": total_return,
            "cagr": cagr_value,
            "max_drawdown": mdd,
            "mdd": mdd,
            "mar": cagr_value / abs(mdd) if mdd else None,
            "trade_count": int(trades.sum()),
            "allocation_change_count": allocation_change_count,
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
    input_batch_hash = stable_json_hash(raw_strategies)
    input_strategy_sequence_hash = strategy_sequence_hash(raw_strategies)
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
    executed_count = sum(1 for row in results if row.get("status") == "executed")
    held_count = len(results) - executed_count
    public_validations = sanitize_public_text(validations)
    public_results = sanitize_public_text(results)
    result_payload = {
        "schema_name": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "input_schema_name": CANONICAL_SCHEMA_NAME,
        "input_schema_version": CANONICAL_SCHEMA_VERSION,
        "run_id": current_run_id,
        "request_id": clean_text(payload.get("request_id")) or current_run_id,
        "generated_at": utc_now(),
        "mode": mode,
        "input_batch_hash": input_batch_hash,
        "input_strategy_sequence_hash": input_strategy_sequence_hash,
        "strategy_count": len(raw_strategies),
        "request_batch_hash": clean_text(payload.get("input_batch_hash")),
        "request_strategy_sequence_hash": clean_text(payload.get("input_strategy_sequence_hash")),
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
        "backtest_summary": {
            "total_rows": len(results),
            "executed_rows": executed_count if mode == "backtest" else 0,
            "held_or_skipped_rows": held_count if mode == "backtest" else len(results),
            "status_counts": {
                "executed": executed_count,
                "skipped": held_count,
            },
            "hold_reason_counts": reason_counts,
            "metrics_available": mode == "backtest" and executed_count > 0,
        },
        "validations": public_validations,
        "results": public_results,
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
        "request_id": result_payload["request_id"],
        "path": f"external_strategy_results/runs/{current_run_id}.json",
        "generated_at": result_payload["generated_at"],
        "mode": mode,
        "input_batch_hash": input_batch_hash,
        "input_strategy_sequence_hash": input_strategy_sequence_hash,
        "strategy_count": len(raw_strategies),
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

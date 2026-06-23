# multi_asset/agent.py — Gemini agent that scores global asset classes.
#
# Usage:
#   python -m multi_asset.agent                  # Gemini + RAG (1 API call)
#   python -m multi_asset.agent --mock           # momentum proxy, no API key
#   python -m multi_asset.agent --model gemini-2.0-flash
#   python -m multi_asset.agent --per-category   # 4 API calls (more quota)
#   python multi_asset/agent.py --mock         # direct script (also works)

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    ASSET_CLASSES_FILE,
    ENV_FILE,
    GEMINI_MODEL,
    MULTI_ASSET_CONTEXT_LOOKBACK,
    MULTI_ASSET_NEWS_INDEX_DIR,
    MULTI_ASSET_SCORES_DIR,
    NEWS_LOOKBACK_DAYS,
)
from env_loader import load_env_file
from multi_asset.universe import (
    asset_classes_by_category,
    list_asset_classes,
    list_categories,
    load_asset_classes,
    proxy_map,
)

########################################################
# Market context
########################################################


def fetch_asset_context(
    asset_classes: list[str] | None = None,
    as_of: str | date | None = None,
    lookback: int = MULTI_ASSET_CONTEXT_LOOKBACK,
    classes_file: str = ASSET_CLASSES_FILE,
) -> dict[str, dict]:
    """Recent ETF/FX performance per asset class as grounding context."""
    df = load_asset_classes(classes_file)
    if asset_classes is not None:
        df = df[df["asset_class"].isin(asset_classes)]

    as_of_ts = pd.Timestamp(as_of or date.today())
    start = (as_of_ts - pd.Timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
    end = (as_of_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    symbols = sorted(df["proxy_symbol"].unique().tolist())
    hist = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
    if hist.empty:
        return {
            row["asset_class"]: {
                "category": row["category"],
                "proxy": row["proxy_symbol"],
                "proxy_type": row["proxy_type"],
                "return_1m": None,
                "return_3m": None,
            }
            for _, row in df.iterrows()
        }

    close = hist["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(symbols[0])

    context: dict[str, dict] = {}
    for _, row in df.iterrows():
        ac = row["asset_class"]
        sym = row["proxy_symbol"]
        base = {
            "category": row["category"],
            "proxy": sym,
            "proxy_type": row["proxy_type"],
        }

        if sym not in close.columns:
            context[ac] = {**base, "return_1m": None, "return_3m": None}
            continue

        series = close[sym].dropna()
        series = series[series.index <= as_of_ts]
        if len(series) < 22:
            context[ac] = {**base, "return_1m": None, "return_3m": None}
            continue

        ret_1m = float(series.iloc[-1] / series.iloc[-21] - 1) if len(series) >= 21 else None
        ret_3m = float(
            series.iloc[-1] / series.iloc[-min(lookback, len(series))] - 1
        )

        context[ac] = {
            **base,
            "return_1m": round(ret_1m, 4) if ret_1m is not None else None,
            "return_3m": round(ret_3m, 4),
        }

    return context


def fetch_news_rag_context(
    as_of: str,
    asset_classes: list[str] | None = None,
    *,
    rebuild: bool = False,
) -> dict[str, list[dict]]:
    from multi_asset.rag import (
        build_and_save_index,
        load_index,
        retrieve_all_asset_classes,
    )

    index_dir = Path(MULTI_ASSET_NEWS_INDEX_DIR) / as_of
    if index_dir.is_dir() and not rebuild:
        print("Loading cached multi-asset news index...")
        index = load_index(as_of)
    else:
        print("Ingesting macro news and building multi-asset RAG index...")
        index, _ = build_and_save_index(as_of=as_of, lookback_days=NEWS_LOOKBACK_DAYS)

    queries = {k: v["rag_query"] for k, v in proxy_map().items()}
    classes = asset_classes or list_asset_classes()
    return retrieve_all_asset_classes(index, asset_classes=classes, rag_queries=queries)


########################################################
# LLM scoring
########################################################

SYSTEM_PROMPT = """You are a systematic multi-asset strategist scoring asset classes.

Given ETF/FX proxy performance and macro news headlines, assign cross-sectional
attractiveness scores for the next 1-3 months WITHIN THE GIVEN CATEGORY ONLY.

Rules:
- Output ONLY valid JSON, no markdown or extra text.
- Scores must be floats in [-1.0, 1.0].
- +1.0 = strongly overweight vs other asset classes in this category.
- -1.0 = strongly underweight.
- 0.0 = neutral within the category.
- Consider macro regime (rates, inflation, growth, risk appetite, FX), not just price momentum.
- If strong recent gains are already priced in per the news, score more cautiously.
- Scores should be differentiated; avoid giving every asset class the same number."""


def _append_class_context(
    lines: list[str],
    asset_classes: list[str],
    context: dict[str, dict],
    news_hits: dict[str, list[dict]] | None,
) -> None:
    lines.append("Asset classes and recent proxy performance:")
    for ac in asset_classes:
        ctx = context.get(ac, {})
        proxy = ctx.get("proxy") or "N/A"
        ptype = ctx.get("proxy_type") or "etf"
        r1 = ctx.get("return_1m")
        r3 = ctx.get("return_3m")
        r1_s = f"{r1:+.2%}" if r1 is not None else "N/A"
        r3_s = f"{r3:+.2%}" if r3 is not None else "N/A"
        lines.append(f"  - {ac} ({ptype} {proxy}): 1m={r1_s}, 3m={r3_s}")

    if news_hits is not None:
        lines.extend(["", "Relevant macro headlines (local RAG top-k per asset class):"])
        for ac in asset_classes:
            hits = news_hits.get(ac, [])
            lines.append(f"  {ac}:")
            if not hits:
                lines.append("    (no headlines in lookback window)")
                continue
            for hit in hits:
                proxy = hit.get("proxy", "?")
                title = hit.get("title", "")
                publisher = hit.get("publisher") or ""
                pub = f" — {publisher}" if publisher else ""
                lines.append(f"    - [{proxy}] {title}{pub}")


def build_user_prompt(
    category: str,
    asset_classes: list[str],
    context: dict[str, dict],
    as_of: str,
    news_hits: dict[str, list[dict]] | None = None,
) -> str:
    lines = [
        f"As-of date: {as_of}",
        f"Category: {category}",
        "",
    ]
    _append_class_context(lines, asset_classes, context, news_hits)
    lines.extend([
        "",
        "Return JSON exactly in this shape:",
        '{"as_of": "YYYY-MM-DD", "category": "CategoryName", "scores": {"AssetClass": 0.0}}',
        f"Include all asset classes in this category: {asset_classes}",
    ])
    return "\n".join(lines)


def build_full_prompt(
    by_category: dict[str, list[str]],
    context: dict[str, dict],
    as_of: str,
    news_hits: dict[str, list[dict]] | None = None,
) -> str:
    """Single prompt covering all categories (one API call)."""
    all_classes = [ac for classes in by_category.values() for ac in classes]
    lines = [
        f"As-of date: {as_of}",
        "",
        "Score each asset class relative to others IN THE SAME CATEGORY ONLY.",
        "",
    ]
    for category in sorted(by_category.keys()):
        classes = by_category[category]
        lines.extend([f"### Category: {category}", ""])
        _append_class_context(lines, classes, context, news_hits)
        lines.append("")

    lines.extend([
        "Return JSON exactly in this shape:",
        '{"as_of": "YYYY-MM-DD", "scores": {"AssetClass": 0.0}}',
        f"Include all {len(all_classes)} asset classes: {all_classes}",
    ])
    return "\n".join(lines)


def _parse_scores(raw: str, asset_classes: list[str]) -> dict[str, float]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]

    data = json.loads(text)
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError("LLM response missing 'scores' object")

    out: dict[str, float] = {}
    for ac in asset_classes:
        if ac not in scores:
            raise ValueError(f"Missing score for asset class: {ac}")
        val = float(scores[ac])
        out[ac] = float(np.clip(val, -1.0, 1.0))
    return out


def _is_daily_quota_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "PerDay" in msg or "per day" in msg.lower() or "FreeTier" in msg


def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return min(60.0, 6.0 * (2 ** attempt))


def _gemini_generate(
    prompt: str,
    model: str = GEMINI_MODEL,
    *,
    max_retries: int = 5,
) -> str:
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"GEMINI_API_KEY not set. Add it to {ENV_FILE} (see .env.example) "
            "or export it, or run with --mock."
        )

    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
        response_mime_type="application/json",
    )

    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            last_exc = exc
            code = getattr(exc, "code", None) or getattr(exc, "status_code", "?")
            if _is_daily_quota_error(exc):
                raise RuntimeError(
                    f"Gemini daily quota exhausted for model '{model}' (free tier ~20 req/day). "
                    "Try: (1) wait until tomorrow, (2) --model gemini-2.0-flash, "
                    "(3) python multi_asset/agent.py --mock"
                ) from exc
            if attempt + 1 >= max_retries:
                break
            delay = _retry_delay_seconds(exc, attempt)
            print(f"Gemini API {code}, retry in {delay:.0f}s ({attempt + 1}/{max_retries})...")
            time.sleep(delay)

    raise RuntimeError(f"Gemini API failed after {max_retries} attempts") from last_exc


def score_category_gemini(
    category: str,
    asset_classes: list[str],
    context: dict[str, dict],
    as_of: str,
    model: str = GEMINI_MODEL,
    news_hits: dict[str, list[dict]] | None = None,
) -> dict[str, float]:
    prompt = build_user_prompt(category, asset_classes, context, as_of, news_hits=news_hits)
    raw = _gemini_generate(prompt, model=model)
    return _parse_scores(raw, asset_classes)


def score_all_gemini(
    context: dict[str, dict],
    as_of: str,
    model: str = GEMINI_MODEL,
    news_hits: dict[str, list[dict]] | None = None,
    by_category: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Score all asset classes in a single Gemini call (saves API quota)."""
    by_category = by_category or asset_classes_by_category()
    all_classes = list_asset_classes()
    prompt = build_full_prompt(by_category, context, as_of, news_hits=news_hits)
    raw = _gemini_generate(prompt, model=model)
    return _parse_scores(raw, all_classes)


def score_category_mock(
    asset_classes: list[str],
    context: dict[str, dict],
) -> dict[str, float]:
    """3-month proxy momentum z-scored to [-1, 1] within the category."""
    raw_returns = []
    for ac in asset_classes:
        r3 = context.get(ac, {}).get("return_3m")
        raw_returns.append(r3 if r3 is not None else 0.0)

    arr = np.array(raw_returns, dtype=float)
    if arr.std() < 1e-8:
        return {ac: 0.0 for ac in asset_classes}

    z = (arr - arr.mean()) / arr.std()
    z = np.clip(z / 2.0, -1.0, 1.0)
    return {ac: float(score) for ac, score in zip(asset_classes, z)}


def score_asset_classes_gemini(
    context: dict[str, dict],
    as_of: str,
    model: str = GEMINI_MODEL,
    news_hits: dict[str, list[dict]] | None = None,
    by_category: dict[str, list[str]] | None = None,
    *,
    per_category: bool = False,
) -> dict[str, float]:
    if per_category:
        by_category = by_category or asset_classes_by_category()
        scores: dict[str, float] = {}
        for category, classes in by_category.items():
            cat_scores = score_category_gemini(
                category, classes, context, as_of, model=model, news_hits=news_hits
            )
            scores.update(cat_scores)
        return scores

    return score_all_gemini(
        context, as_of, model=model, news_hits=news_hits, by_category=by_category
    )


def score_asset_classes_mock(
    context: dict[str, dict],
    by_category: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    by_category = by_category or asset_classes_by_category()
    scores: dict[str, float] = {}
    for classes in by_category.values():
        scores.update(score_category_mock(classes, context))
    return scores


########################################################
# Public API
########################################################


def score_asset_classes(
    as_of: str | date | None = None,
    *,
    use_llm: bool = True,
    use_news: bool = True,
    save: bool = True,
) -> dict[str, float]:
    """
    Score all asset classes for a given date.

    Scores are cross-sectional within each category (Equities, Fixed_Income, …).
    Returns dict mapping asset_class -> score in [-1, 1].
    """
    as_of_str = (as_of or date.today()).strftime("%Y-%m-%d") if not isinstance(as_of, str) else as_of
    classes = list_asset_classes()
    by_cat = asset_classes_by_category()
    context = fetch_asset_context(classes, as_of_str)

    news_hits: dict[str, list[dict]] | None = None
    if use_news and use_llm:
        news_hits = fetch_news_rag_context(as_of_str, classes)

    if use_llm:
        scores = score_asset_classes_gemini(context, as_of_str, news_hits=news_hits, by_category=by_cat)
        source = "gemini_rag" if use_news else "gemini"
    else:
        scores = score_asset_classes_mock(context, by_category=by_cat)
        source = "mock"

    if save:
        save_asset_scores(
            as_of_str,
            scores,
            context,
            source=source,
            news_sources=news_hits,
            by_category=by_cat,
        )

    return scores


def save_asset_scores(
    as_of: str,
    scores: dict[str, float],
    context: dict[str, dict] | None = None,
    source: str = "llm",
    news_sources: dict[str, list[dict]] | None = None,
    by_category: dict[str, list[str]] | None = None,
) -> Path:
    out_dir = Path(MULTI_ASSET_SCORES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{as_of}.json"

    by_category = by_category or asset_classes_by_category()
    scores_by_category = {
        cat: {ac: scores[ac] for ac in classes if ac in scores}
        for cat, classes in by_category.items()
    }

    payload = {
        "as_of": as_of,
        "source": source,
        "scores": scores,
        "scores_by_category": scores_by_category,
        "context": context or {},
    }
    if news_sources is not None:
        payload["news_sources"] = news_sources
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_asset_scores(as_of: str) -> dict[str, float]:
    path = Path(MULTI_ASSET_SCORES_DIR) / f"{as_of}.json"
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data["scores"].items()}


def print_scores(
    scores: dict[str, float],
    as_of: str,
    source: str,
    by_category: dict[str, list[str]] | None = None,
) -> None:
    by_category = by_category or asset_classes_by_category()
    print(f"\n--- Multi-asset scores ({source}) as of {as_of} ---")
    for category in sorted(by_category.keys()):
        classes = by_category[category]
        print(f"\n  [{category}]")
        ranked = sorted(
            ((ac, scores[ac]) for ac in classes if ac in scores),
            key=lambda x: x[1],
            reverse=True,
        )
        for ac, score in ranked:
            bar = "+" * int(max(0, score) * 20) or "-" * int(abs(min(0, score)) * 20)
            print(f"    {ac:24s}  {score:+.3f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score global asset classes via Gemini agent")
    parser.add_argument("--date", default=None, help="As-of date YYYY-MM-DD (default: today)")
    parser.add_argument("--mock", action="store_true", help="Use proxy momentum (no API key)")
    parser.add_argument("--no-news", action="store_true", help="Skip RAG news (market context only)")
    parser.add_argument("--rebuild-news", action="store_true", help="Rebuild news index even if cached")
    parser.add_argument("--no-save", action="store_true", help="Do not write scores JSON")
    parser.add_argument("--model", default=GEMINI_MODEL, help=f"Gemini model (default: {GEMINI_MODEL})")
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="One API call per category (uses more quota; default is single call)",
    )
    args = parser.parse_args()

    as_of = args.date or date.today().strftime("%Y-%m-%d")
    by_cat = asset_classes_by_category()
    n_classes = sum(len(v) for v in by_cat.values())
    print(f"Asset classes ({n_classes}) across {len(by_cat)} categories:")
    for cat, classes in by_cat.items():
        print(f"  {cat}: {len(classes)}")
    print("Fetching ETF/FX context...")

    classes = list_asset_classes()
    context = fetch_asset_context(classes, as_of)
    use_llm = not args.mock
    use_news = not args.no_news and use_llm

    news_hits: dict[str, list[dict]] | None = None
    if use_news:
        news_hits = fetch_news_rag_context(as_of, classes, rebuild=args.rebuild_news)
        n_articles = sum(len(v) for v in news_hits.values())
        print(f"RAG retrieved {n_articles} headlines across {len(classes)} asset classes")

    if use_llm:
        source = "gemini_rag" if use_news else "gemini"
        n_calls = len(by_cat) if args.per_category else 1
        print(f"Calling Gemini ({args.model}, {n_calls} API call{'s' if n_calls > 1 else ''})...")
        scores = score_asset_classes_gemini(
            context, as_of, model=args.model, news_hits=news_hits,
            by_category=by_cat, per_category=args.per_category,
        )
        if not args.no_save:
            save_asset_scores(as_of, scores, context, source=source, news_sources=news_hits, by_category=by_cat)
    else:
        scores = score_asset_classes_mock(context, by_category=by_cat)
        if not args.no_save:
            save_asset_scores(as_of, scores, context, source="mock", by_category=by_cat)

    print_scores(scores, as_of, source if use_llm else "mock", by_category=by_cat)
    if not args.no_save:
        print(f"\nSaved to {MULTI_ASSET_SCORES_DIR}/{as_of}.json")


if __name__ == "__main__":
    main()

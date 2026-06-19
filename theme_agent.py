# theme_agent.py — Gemini agent that scores investment themes for the AI factor pipeline.
#
# Usage:
#   python theme_agent.py                  # Gemini + RAG news + ETF context
#   python theme_agent.py --mock           # ETF-momentum proxy, no API key
#   python theme_agent.py --no-news        # Gemini + ETF only, skip RAG
#   python theme_agent.py --date 2026-06-16

import argparse
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    THEMES_FILE,
    THEME_ETF_PROXY,
    THEME_CONTEXT_LOOKBACK,
    GEMINI_MODEL,
    THEME_SCORES_DIR,
    ENV_FILE,
    NEWS_INDEX_DIR,
    NEWS_LOOKBACK_DAYS,
)
from env_loader import load_env_file

########################################################
# Theme map & market context
########################################################


def load_theme_map(path: str = THEMES_FILE) -> pd.DataFrame:
    """Load stock-to-theme weights from CSV."""
    df = pd.read_csv(path)
    required = {"ticker", "theme", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must have columns: {sorted(required)}")
    return df


def list_themes(path: str = THEMES_FILE) -> list[str]:
    """Unique theme names in themes.csv."""
    return sorted(load_theme_map(path)["theme"].unique().tolist())


def fetch_theme_context(
    themes: list[str],
    as_of: str | date,
    lookback: int = THEME_CONTEXT_LOOKBACK,
) -> dict[str, dict]:
    """Recent ETF performance per theme as grounding context for the agent."""
    as_of_ts = pd.Timestamp(as_of)
    start = (as_of_ts - pd.Timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
    end = (as_of_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    context: dict[str, dict] = {}
    for theme in themes:
        etf = THEME_ETF_PROXY.get(theme)
        if not etf:
            context[theme] = {"etf": None, "return_1m": None, "return_3m": None}
            continue

        hist = yf.download(etf, start=start, end=end, progress=False, auto_adjust=True)
        if hist.empty or "Close" not in hist.columns:
            context[theme] = {"etf": etf, "return_1m": None, "return_3m": None}
            continue

        close = hist["Close"].dropna()
        if isinstance(close, pd.DataFrame):
            close = close.squeeze(axis=1)
        close = close[close.index <= as_of_ts]
        if len(close) < 22:
            context[theme] = {"etf": etf, "return_1m": None, "return_3m": None}
            continue

        ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else None
        ret_3m = float(close.iloc[-1] / close.iloc[-min(lookback, len(close))] - 1)

        context[theme] = {
            "etf": etf,
            "return_1m": round(ret_1m, 4) if ret_1m is not None else None,
            "return_3m": round(ret_3m, 4),
        }

    return context


def fetch_news_rag_context(
    themes: list[str],
    as_of: str,
    *,
    rebuild: bool = False,
) -> dict[str, list[dict]]:
    """Build or load local RAG index and retrieve top-k news per theme."""
    from news.rag import build_and_save_index, load_index, retrieve_all_themes

    index_dir = Path(NEWS_INDEX_DIR) / as_of
    if index_dir.is_dir() and not rebuild:
        print("Loading cached news index...")
        index = load_index(as_of)
    else:
        print("Ingesting news and building local RAG index...")
        index, _ = build_and_save_index(as_of=as_of, lookback_days=NEWS_LOOKBACK_DAYS)

    return retrieve_all_themes(index, themes=themes)


########################################################
# LLM scoring
########################################################

SYSTEM_PROMPT = """You are a systematic equity research analyst scoring investment themes.

Given ETF proxy performance and RAG-retrieved news headlines for each theme, assign a cross-sectional attractiveness score.

Rules:
- Output ONLY valid JSON, no markdown or extra text.
- Scores must be floats in [-1.0, 1.0].
- +1.0 = strongly bullish for the next 1-3 months relative to other themes.
- -1.0 = strongly bearish.
- 0.0 = neutral.
- Use ETF momentum AND news sentiment/outlook; do not blindly follow recent price moves.
- If strong recent ETF gains are already in the news as "priced in", score more cautiously.
- Scores should be differentiated; avoid giving every theme the same number."""


def build_user_prompt(
    themes: list[str],
    context: dict[str, dict],
    as_of: str,
    news_hits: dict[str, list[dict]] | None = None,
) -> str:
    lines = [
        f"As-of date: {as_of}",
        "",
        "Themes and recent ETF proxy performance:",
    ]
    for theme in themes:
        ctx = context.get(theme, {})
        etf = ctx.get("etf") or "N/A"
        r1 = ctx.get("return_1m")
        r3 = ctx.get("return_3m")
        r1_s = f"{r1:+.2%}" if r1 is not None else "N/A"
        r3_s = f"{r3:+.2%}" if r3 is not None else "N/A"
        lines.append(f"  - {theme} (proxy {etf}): 1m={r1_s}, 3m={r3_s}")

    if news_hits is not None:
        lines.extend(["", "Relevant news headlines (local RAG top-k per theme):"])
        for theme in themes:
            hits = news_hits.get(theme, [])
            lines.append(f"  {theme}:")
            if not hits:
                lines.append("    (no headlines in lookback window)")
                continue
            for hit in hits:
                ticker = hit.get("ticker", "?")
                title = hit.get("title", "")
                publisher = hit.get("publisher") or ""
                pub = f" — {publisher}" if publisher else ""
                lines.append(f"    - [{ticker}] {title}{pub}")

    lines.extend([
        "",
        "Return JSON exactly in this shape:",
        '{"as_of": "YYYY-MM-DD", "scores": {"ThemeName": 0.0}}',
        f'Include all themes: {themes}',
    ])
    return "\n".join(lines)


def _parse_scores(raw: str, themes: list[str]) -> dict[str, float]:
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
    for theme in themes:
        if theme not in scores:
            raise ValueError(f"Missing score for theme: {theme}")
        val = float(scores[theme])
        out[theme] = float(np.clip(val, -1.0, 1.0))
    return out


def score_themes_gemini(
    themes: list[str],
    context: dict[str, dict],
    as_of: str,
    model: str = GEMINI_MODEL,
    news_hits: dict[str, list[dict]] | None = None,
) -> dict[str, float]:
    """Call Google Gemini to score themes. Key from .env or GEMINI_API_KEY env var."""
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"GEMINI_API_KEY not set. Add it to {ENV_FILE} (see .env.example) "
            "or export it, or run with --mock."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=build_user_prompt(themes, context, as_of, news_hits=news_hits),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    return _parse_scores(raw, themes)


def score_themes_mock(
    themes: list[str],
    context: dict[str, dict],
) -> dict[str, float]:
    """ETF 3-month momentum proxy, z-scored to [-1, 1]. No API key needed."""
    raw_returns = []
    for theme in themes:
        r3 = context.get(theme, {}).get("return_3m")
        raw_returns.append(r3 if r3 is not None else 0.0)

    arr = np.array(raw_returns, dtype=float)
    if arr.std() < 1e-8:
        return {t: 0.0 for t in themes}

    z = (arr - arr.mean()) / arr.std()
    z = np.clip(z / 2.0, -1.0, 1.0)
    return {theme: float(score) for theme, score in zip(themes, z)}


########################################################
# Public API
########################################################


def score_themes(
    as_of: str | date | None = None,
    *,
    use_llm: bool = True,
    use_news: bool = True,
    themes_file: str = THEMES_FILE,
    save: bool = True,
) -> dict[str, float]:
    """
    Score all themes for a given date.

    Returns dict mapping theme name -> score in [-1, 1].
    """
    as_of_str = (as_of or date.today()).strftime("%Y-%m-%d") if not isinstance(as_of, str) else as_of
    themes = list_themes(themes_file)
    context = fetch_theme_context(themes, as_of_str)

    news_hits: dict[str, list[dict]] | None = None
    if use_news and use_llm:
        news_hits = fetch_news_rag_context(themes, as_of_str)

    if use_llm:
        scores = score_themes_gemini(themes, context, as_of_str, news_hits=news_hits)
        source = "gemini_rag" if use_news else "gemini"
    else:
        scores = score_themes_mock(themes, context)
        source = "mock"

    if save:
        save_theme_scores(
            as_of_str, scores, context,
            source=source,
            news_sources=news_hits,
        )

    return scores


def save_theme_scores(
    as_of: str,
    scores: dict[str, float],
    context: dict[str, dict] | None = None,
    source: str = "llm",
    news_sources: dict[str, list[dict]] | None = None,
) -> Path:
    out_dir = Path(THEME_SCORES_DIR)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{as_of}.json"
    payload = {
        "as_of": as_of,
        "source": source,
        "scores": scores,
        "context": context or {},
    }
    if news_sources is not None:
        payload["news_sources"] = news_sources
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_theme_scores(as_of: str) -> dict[str, float]:
    """Load previously saved scores for a date."""
    path = Path(THEME_SCORES_DIR) / f"{as_of}.json"
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data["scores"].items()}


def print_scores(scores: dict[str, float], as_of: str, source: str) -> None:
    print(f"\n--- Theme scores ({source}) as of {as_of} ---")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for theme, score in ranked:
        bar = "+" * int(max(0, score) * 20) or "-" * int(abs(min(0, score)) * 20)
        print(f"  {theme:16s}  {score:+.3f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score investment themes via Gemini agent")
    parser.add_argument("--date", default=None, help="As-of date YYYY-MM-DD (default: today)")
    parser.add_argument("--mock", action="store_true", help="Use ETF momentum proxy (no API key)")
    parser.add_argument("--no-news", action="store_true", help="Skip RAG news (ETF context only)")
    parser.add_argument("--rebuild-news", action="store_true", help="Rebuild news index even if cached")
    parser.add_argument("--no-save", action="store_true", help="Do not write theme_scores/ JSON")
    args = parser.parse_args()

    as_of = args.date or date.today().strftime("%Y-%m-%d")
    themes = list_themes()
    print(f"Themes ({len(themes)}): {', '.join(themes)}")
    print("Fetching ETF context...")

    context = fetch_theme_context(themes, as_of)
    use_llm = not args.mock
    use_news = not args.no_news and use_llm

    news_hits: dict[str, list[dict]] | None = None
    if use_news:
        news_hits = fetch_news_rag_context(themes, as_of, rebuild=args.rebuild_news)
        n_articles = sum(len(v) for v in news_hits.values())
        print(f"RAG retrieved {n_articles} headlines across {len(themes)} themes")

    if use_llm:
        source = "gemini_rag" if use_news else "gemini"
        scores = score_themes_gemini(themes, context, as_of, news_hits=news_hits)
        if not args.no_save:
            save_theme_scores(as_of, scores, context, source=source, news_sources=news_hits)
    else:
        scores = score_themes_mock(themes, context)
        if not args.no_save:
            save_theme_scores(as_of, scores, context, source="mock")

    print_scores(scores, as_of, source if use_llm else "mock")
    if not args.no_save:
        print(f"\nSaved to {THEME_SCORES_DIR}/{as_of}.json")


if __name__ == "__main__":
    main()

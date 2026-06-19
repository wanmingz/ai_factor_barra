# AI Theme Factor Research

**Can LLM + RAG-enhanced thematic views improve cross-sectional equity return explanation and stock ranking, beyond classical Barra style factors?**

This repo is a **research sandbox** for studying the **AI Theme Factor** — a 5th factor derived from Gemini theme scores (grounded in ETF momentum and RAG-retrieved news) and stock-theme mappings, embedded into a Barra-style multi-factor framework alongside Size, Value, Momentum, and Volatility.

The classical factor pipeline (WLS → ML → IC → long-short backtest) provides the **measurement infrastructure**. The research contribution is testing whether **AI-driven thematic signals** add incremental explanatory and predictive power.

> **Scope**: Educational / portfolio project. Not a production risk model. PnL figures are illustrative; research design and limitations are documented explicitly.

---

## Core Research Question

> *When an LLM agent scores investment themes (AI, Semiconductors, Energy, …) using ETF context and RAG-retrieved news, does mapping those scores to individual stocks — via thematic exposure weights — produce a factor that (a) earns a positive risk premium in WLS, (b) is orthogonal to Momentum, and (c) improves out-of-sample Rank IC when added to a 4-factor model?*

### Sub-questions

| # | Question | How we test it |
|---|----------|----------------|
| H1 | Does the AI Theme Factor have a positive **factor premium** (β_AI)? | 5-factor WLS: `r_i − r_SPY = X·β + ε` |
| H2 | Is it **orthogonal** to classical factors? | Factor exposure correlation matrix; especially vs. Momentum |
| H3 | Does it add **incremental R²** over 4 factors? | Compare cross-sectional R²: 4-factor vs. 5-factor WLS |
| H4 | Does it improve **stock ranking** (Rank IC)? | Ridge 4-factor vs. Ridge 5-factor on forward excess return |
| H5 | Does **Gemini + RAG** beat a naive **ETF-momentum** baseline? | Compare `theme_agent.py` vs. `--mock` scores |

---

## The AI Theme Factor

### Definition

**Step 1 — Theme scores** (daily, cross-sectional view on themes):

```
ThemeScore(t, k) ∈ [-1, +1]     # Gemini + RAG + ETF context, or --mock proxy
```

**Step 2 — Stock thematic exposure** (from `themes.csv`):

```
Raw_AI(t, i) = Σ_k  weight(i, k) × ThemeScore(t, k)
```

Each stock can map to multiple themes with weights summing to 1. Example: NVDA → 55% AI + 45% Semiconductors.

**Step 3 — Cross-sectional standardization** (same as style factors):

```
AI_Exposure(t, i) = z-score_i( Raw_AI(t, ·) )
```

**Step 4 — Embed in Barra WLS** as the 5th column of X:

```
r_i − r_SPY ≈ β_Size·X_Size + β_Value·X_Value + β_Mom·X_Mom + β_Vol·X_Vol + β_AI·X_AI + ε_i
```

A positive, stable **β_AI** means high-AI-exposure stocks earn excess returns beyond what the four style factors explain.

### Why themes + LLM + RAG?

| Layer | Role |
|-------|------|
| **RAG (local)** | Retrieve theme-relevant news headlines — free, no embedding API |
| **Gemini** | Synthesize ETF momentum + news into theme scores |
| **`themes.csv`** | Map theme views to stock exposures |
| **Barra WLS / ML** | Measure incremental premium and Rank IC |

The pipeline separates **information retrieval** (RAG), **view generation** (Gemini), **position mapping** (weights), and **factor estimation** (WLS) — each step is auditable.

---

## Research Architecture

```mermaid
flowchart TB
    subgraph rag [News RAG — free local]
        YFNews[yfinance news] --> Ingest[news/ingest.py]
        Ingest --> Embed["sentence-transformers<br/>all-MiniLM-L6-v2"]
        Embed --> Index[data/news_index]
        Index --> TopK[top-k per theme]
    end

    subgraph agent [Theme Agent]
        ETF[ETF 1m/3m returns] --> GEM[Gemini]
        TopK --> GEM
        GEM --> TS["ThemeScore(t,k)"]
    end

    subgraph factor [AI Factor — next]
        CSV[themes.csv] --> RAW["Raw_AI(t,i)"]
        TS --> RAW
        RAW --> ZS[AI_Exposure z-score]
    end

    subgraph baseline [4 Style Factors]
        PX[Prices] --> F4[Size Value Mom Vol]
        F4 --> X4[z-score]
    end

    subgraph test [Impact Measurement]
        ZS --> X5[5-factor X panel]
        X4 --> X5
        X5 --> WLS["WLS → β_AI"]
        X5 --> ML[Rank IC + backtest]
    end
```

### Implementation status

| Component | Status | Module |
|-----------|--------|--------|
| Stock-theme mapping (50 stocks × 10 themes) | ✅ Done | `themes.csv` |
| yfinance news ingest per theme | ✅ Done | `news/ingest.py` |
| Local RAG (free embedding + retrieval) | ✅ Done | `news/rag.py` |
| Gemini scoring + RAG in prompt | ✅ Done | `theme_agent.py` |
| ETF-momentum baseline (`--mock`) | ✅ Done | `theme_agent.py` |
| Daily score + news citation persistence | ✅ Done | `theme_scores/*.json` |
| Stock AI exposure panel |  ✅ Done | `ai_factor.py` |
| 5-factor WLS (β_AI, ΔR²) | 🔲 Next | `barra_panel.py` |
| 5-factor ML (incremental Rank IC) | 🔲 Next | `ml_predict.py` |
| Historical theme score panel (mock batch) | 🔲 Next | `ai_factor.py` |

---

## News RAG Pipeline (free — no paid embedding API)

### `news/ingest.py` — collect headlines

1. Look up tickers per theme from `themes.csv` (e.g. AI → NVDA, MSFT, AMD…)
2. Fetch recent headlines via `yfinance.Ticker(t).news`
3. Deduplicate by title; filter to `NEWS_LOOKBACK_DAYS` (default 7) before `as_of`

### `news/rag.py` — retrieve top-k per theme

1. Embed headlines with **local** `sentence-transformers` (`all-MiniLM-L6-v2`)
2. Query each theme: `"{theme} sector investment outlook stock market news"`
3. Cosine similarity → top-`NEWS_RAG_TOP_K` (default 5) articles per theme
4. Cache index under `data/news_index/YYYY-MM-DD/` (JSON + `.npy` vectors)

```bash
python news/ingest.py    # smoke test: fetch headlines
python news/rag.py       # build index + print retrieval results
```

**Cost**: embedding runs locally after first model download — no OpenAI/Gemini embedding API.

**Limitation**: yfinance news has no full historical archive; RAG is for **live / recent** scoring. Historical backtests use `--mock` ETF proxy scores.

---

## Theme Agent (`theme_agent.py`)

Combines **ETF context**, **RAG news**, and **Gemini** to produce daily theme scores.

### Workflow

```mermaid
flowchart LR
    CSV[themes.csv] --> Ingest
    YF[yfinance news] --> Ingest[ingest + RAG]
    Ingest --> Prompt[Gemini prompt]
    ETF[BOTZ SMH XLE …] --> Prompt
    Prompt --> JSON["theme_scores/*.json"]
```

1. Fetch ETF proxy returns (1m / 3m) per theme
2. Build or load cached RAG index; retrieve top-k news per theme
3. Prompt Gemini with ETF data + headlines
4. Save JSON with `scores`, `context`, and `news_sources`

### CLI

```bash
python theme_agent.py                  # Gemini + RAG + ETF (default)
python theme_agent.py --date 2026-06-19
python theme_agent.py --no-news        # Gemini + ETF only (A/B vs RAG)
python theme_agent.py --rebuild-news   # force rebuild news index
python theme_agent.py --mock           # ETF proxy baseline, no API key
```

Requires `GEMINI_API_KEY` in `.env` (see `.env.example`). Use API model ID in `config.py` (e.g. `gemini-2.5-flash-lite`), not the UI display name.

### Example output (`theme_scores/2026-06-19.json`, source: `gemini_rag`)

| Theme | Score | Notes |
|-------|-------|-------|
| Semiconductors | **+0.80** | Bullish; strong SMH momentum |
| Industrials | +0.60 | |
| Energy | **−0.60** | Bearish; weak XLE |
| Media | −0.40 | |
| Software | 0.00 | Neutral despite positive ETF — Gemini ≠ pure momentum |

JSON also includes `news_sources` with cited headlines per theme for interview demos.

### Baseline for H5

`--mock` z-scores ETF 3-month returns across themes → [-1, +1]. If Gemini+RAG cannot beat mock on Rank IC or β_AI stability, the LLM layer adds no value over naive theme momentum.

---

## Getting AI Factor into Barra (next step)

Theme scores exist; **`ai_factor.py` is not yet implemented**. The intended integration:

```python
# 1. Load theme scores for date t
scores = load_theme_scores("2026-06-19")

# 2. Map to stocks via themes.csv
Raw_AI(i) = Σ weight(i, theme) × scores[theme]

# 3. Merge with 4 style factors, z-score, WLS
FACTOR_NAMES = ["Size", "Value", "Momentum", "Volatility", "AI"]
```

See [`barra_panel.py`](barra_panel.py): once `X_panel` has 5 columns, WLS automatically reports `AI_Return` as the 5th β.

---

## Foundation Pipeline

The 4-factor Barra + ML stack is the **control framework** against which AI Theme Factor impact is measured.

### Style factors (4)

| Factor | Construction |
|--------|--------------|
| Size | `ln(price)` |
| Value | `book_per_share / price` (P/B proxy) |
| Momentum | 63d cumulative return, skip last 5d |
| Volatility | 20d rolling return std |

### WLS factor returns

```
β = (Xᵀ W X)⁻¹ Xᵀ W y        W = diag(ln(marketCap))
```

### ML alpha (4-factor baseline results)

Chronological 80/20 split, 50 large caps, 2016–2026:

```
                          mean_ic  mean_rank_ic  sharpe
Ridge (4 factors)          0.0730        0.0546    1.95
Momentum only (OLS)        0.0246        0.0216    0.80
Baseline (predict 0)          NaN           NaN   -1.86
```

**This is the bar the 5-factor model must beat.**

### Entry points

| Script | Role |
|--------|------|
| `news/ingest.py` | Fetch yfinance news per theme |
| `news/rag.py` | Local embedding RAG index + retrieval |
| `theme_agent.py` | **Signal** — Gemini + RAG theme scores |
| `barra_panel.py` | Factor panel + WLS (4 factors today) |
| `ml_predict.py` | ML comparison + Rank IC + backtest |
| `barra.py` | Static WLS demo |
| `backtest.py` | Long-short portfolio simulation |

---

## How We Measure AI Theme Factor Impact

Once `ai_factor.py` is integrated:

| Test | Metric |
|------|--------|
| H1 Factor premium | Mean **β_AI** from daily 5-factor WLS |
| H2 Orthogonality | `corr(AI_Exposure, Momentum)` |
| H3 Incremental R² | `R²(5-factor) − R²(4-factor)` per date |
| H4 Predictive power | Rank IC: Ridge(5) vs Ridge(4) on test set |
| H5 LLM value-add | Gemini+RAG vs `--mock` on sample dates |

---

## Project Structure

```
ai_factor_barra/
│
├── config.py                 # Global settings: universe, dates, factor names, RAG/agent params
├── env_loader.py             # Load GEMINI_API_KEY from .env
├── .env.example              # API key template (copy to .env)
├── requirements.txt
├── .gitignore
│
├── themes.csv                # Stock → theme exposure weights (50 stocks × 10 themes)
│
├── theme_agent.py            # Gemini + RAG + ETF → daily theme scores
├── theme_scores/             # Persisted theme score JSON (scores, context, news_sources)
│   └── YYYY-MM-DD.json
│
├── news/                     # Free local news RAG (no paid embedding API)
│   ├── __init__.py
│   ├── ingest.py             # Fetch yfinance headlines per theme
│   └── rag.py                # sentence-transformers embed + cosine retrieval
│
├── ai_factor.py              # Theme scores → stock-level AI exposure panel
├── features.py               # Style factors (Size/Value/Mom/Vol) + cross-section z-score
│
├── barra.py                  # Single-day 5-factor WLS demo (static X from yfinance info)
├── barra_panel.py            # Rolling daily factor panel + 5-factor WLS
├── ml_predict.py             # Ridge/RF models, Rank IC, long-short backtest
├── backtest.py               # Long-short portfolio simulation (used by ml_predict)
│
└── data/                     # Generated caches (gitignored)
    ├── news_index/           # RAG embeddings per as-of date
    │   └── YYYY-MM-DD/
    │       ├── meta.json
    │       ├── AI.json / AI.npy
    │       └── …             # one JSON + .npy per theme
    └── ai_exposure_panel.parquet   # (date, ticker) × AI raw exposure panel
```

### Module roles

| Path | Role |
|------|------|
| `themes.csv` | Static stock–theme weights; input to AI exposure mapping |
| `theme_agent.py` | **Signal layer** — LLM theme scores with RAG + ETF context |
| `theme_scores/` | Auditable archive of daily Gemini/mock scores |
| `news/ingest.py` | Pull and normalize yfinance news by theme |
| `news/rag.py` | Local embedding index + top-k retrieval per theme |
| `ai_factor.py` | Map theme scores → per-stock `Raw_AI`; cache parquet panel |
| `features.py` | Build 4 style factors; merge AI panel; z-score cross-section |
| `barra.py` | Quick single-date WLS including `AI_Return` |
| `barra_panel.py` | Full-history factor panel + WLS factor returns |
| `ml_predict.py` | ML alpha test: Ridge(4) vs Ridge(5), Rank IC, backtest |
| `backtest.py` | Monthly long-short simulation helpers |
| `config.py` | `FACTOR_NAMES`, dates, `AI_SCORE_MODE`, RAG params |
| `data/news_index/` | Cached RAG vectors (rebuilt by `theme_agent` or `news/rag.py`) |
| `data/ai_exposure_panel.parquet` | Cached AI panel for `barra_panel` / `ml_predict` |

### Recommended run order

```bash
# 1. Theme scoring (optional — skip if using --mode mock)
python theme_agent.py --date YYYY-MM-DD

# 2. Build AI exposure panel
python ai_factor.py --mode auto --rebuild    # Gemini JSON where available, else mock
# python ai_factor.py --mode mock --rebuild  # full-history mock baseline

# 3. Factor research (any order)
python barra.py          # single-day WLS snapshot
python barra_panel.py    # rolling panel + WLS
python ml_predict.py     # ML comparison + Rank IC
```

---

## Setup & Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# GEMINI_API_KEY=...  https://aistudio.google.com/apikey

# --- AI Theme signal ---
python news/rag.py                 # build local RAG index (optional standalone test)
python theme_agent.py              # Gemini + RAG + ETF → theme_scores/
python theme_agent.py --mock       # reproducible baseline, no API key

# --- Baseline factor research ---
python barra_panel.py              # 4-factor WLS
python ml_predict.py               # 4-factor ML + Rank IC
```

First run downloads `all-MiniLM-L6-v2` (~80 MB) for local embedding. `ml_predict.py` first run is slow (50 tickers × yfinance info for Value factor).

---

## Configuration

| Parameter | Default | Role |
|-----------|---------|------|
| `THEMES_FILE` | `themes.csv` | Stock-theme exposure weights |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | LLM for theme scoring |
| `THEME_ETF_PROXY` | BOTZ, SMH, … | ETF context in agent prompt |
| `THEME_SCORES_DIR` | `theme_scores` | Daily score archive |
| `NEWS_LOOKBACK_DAYS` | 7 | News date filter |
| `NEWS_RAG_TOP_K` | 5 | Headlines per theme in prompt |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local RAG embedding (free) |
| `NEWS_INDEX_DIR` | `data/news_index` | RAG cache |
| `FACTOR_NAMES` | 4 style factors | Will add `"AI"` after `ai_factor.py` |
| `FORWARD_DAYS` | 5 | ML label horizon |

---

## Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Static `themes.csv` | Business models don't evolve in data | Point-in-time tags in production |
| yfinance news not historical | RAG only for recent/live dates | `--mock` for backtest panel |
| Generic headlines in RAG | "Stocks rally" hits many themes | Theme-specific tickers + semantic query |
| Theme ↔ Momentum overlap | AI factor may duplicate Mom | Report correlation; residualize |
| `ai_factor.py` not built | Cannot run 5-factor WLS yet | Next implementation step |
| 50-stock universe | Noisy β_AI | Focus on methodology in interviews |
| Single ML train/test split | One-regime OOS | Walk-forward as extension |

---

## Interview Talking Points

**Elevator pitch (30 sec):**

> I built a Barra-style research pipeline to test whether LLM thematic views add incremental alpha. A free local RAG layer retrieves theme-relevant news; Gemini scores 10 themes using news + ETF context; stock AI exposure is a weighted sum from `themes.csv`. I measure impact via β_AI, orthogonality to Momentum, ΔR², and incremental Rank IC over a 4-factor baseline.

**Expected questions:**

- **What is the AI Theme Factor?** `Σ weight(i,theme) × ThemeScore(theme)`, z-scored, embedded as a 5th Barra factor — not a black-box stock picker.
- **How does RAG work here?** yfinance headlines → local MiniLM embedding → cosine retrieval per theme → top headlines in Gemini prompt. No paid embedding API.
- **How do you backtest LLM signals?** Live: Gemini+RAG daily snapshots in `theme_scores/`. History: `--mock` ETF proxy batch for reproducible panels.
- **What if AI correlates with Momentum?** Report factor correlation matrix; residualize AI against Momentum if needed.
- **What's next?** `ai_factor.py` to map scores → stock panel, then 5-factor WLS and Ridge(5) vs Ridge(4) Rank IC.

---

## License

Personal / educational use.

# Equity Factor ML

**A cross-sectional factor research pipeline**: Barra-style style factors → WLS factor return estimation → machine-learning alpha prediction → information coefficient (IC) evaluation and long-short backtesting.

Built as a **research sandbox** for exploring how classical factor exposures can be combined with ML to forecast forward excess returns — the kind of end-to-end workflow common in systematic equity research.

> **Scope**: Educational / portfolio project. Not a production risk model. Limitations are documented explicitly below.

> **Roadmap**: A **Gemini theme agent** scores investment themes daily (`theme_agent.py`); stock-level **AI factor** exposure (theme scores × `themes.csv` weights → 5th WLS factor) is the next integration step. See [AI Theme Agent](#ai-theme-agent) and [Planned: AI Factor](#planned-ai-factor).

---

## Research Question

> *Can standardized style-factor exposures predict forward cross-sectional excess returns, and does a simple ML layer improve stock-ranking ability over a single-factor baseline?*

The pipeline answers this in three stages:

1. **Factor model** — Estimate daily factor premia via weighted least squares (WLS).
2. **Alpha model** — Train regressors on factor exposures to predict N-day forward excess return.
3. **Portfolio test** — Translate predictions into a dollar-neutral long-short book and measure IC, Rank IC, and PnL statistics out-of-sample.

A fourth stage — mapping **AI theme scores** to stock exposures and embedding them as a **5th factor** in the Barra WLS framework — is in progress (theme agent done; `ai_factor.py` integration pending).

---

## Architecture

```mermaid
flowchart TB
    subgraph ingest [Data Ingestion]
        YF[Yahoo Finance] --> PX[Prices & Returns]
        PX --> Y["Excess returns Y<br/>r_i − r_SPY"]
    end

    subgraph feat [Feature Engineering]
        PX --> RAW[Raw factor panel]
        RAW --> ZS["Cross-sectional z-score<br/>(date, ticker) × 4 factors"]
    end

    subgraph ai [AI Theme Layer]
        CSV[themes.csv] --> MAP[Stock-theme weights]
        ETF[Theme ETF proxies] --> CTX[Market context]
        CTX --> GEM[Gemini agent]
        GEM --> TS[Theme scores -1 to +1]
        MAP --> AIF[AI exposure panel]
        TS --> AIF
    end

    subgraph est [Estimation]
        ZS --> WLS["WLS: β = (X'WX)⁻¹X'Wy"]
        ZS --> ML["ML: f(X) → ŷ_{t→t+N}"]
        AIF -.-> WLS
    end

    subgraph eval [Evaluation]
        ML --> IC["Daily IC / Rank IC"]
        ML --> LS["Long-short backtest<br/>top 20% / bottom 20%"]
    end
```

### Entry points (increasing complexity)

| Script | Role | Target variable |
|--------|------|-----------------|
| `barra.py` | Static cross-section WLS demo | Same-day excess return |
| `barra_panel.py` | Daily rolling factor panel + WLS on last day | Same-day excess return |
| `ml_predict.py` | Full ML pipeline + model comparison + backtest | Forward N-day excess return |
| `theme_agent.py` | Gemini agent scores investment themes | Theme attractiveness ∈ [-1, 1] |

---

## Project Structure

```
ai_factor_barra/
├── config.py        # Universe, dates, factor windows, ML & backtest params
├── features.py      # Rolling factor computation + cross-sectional z-score
├── barra.py         # Snapshot WLS (quick Barra mechanics demo)
├── barra_panel.py   # Time-series factor panel + WLS
├── ml_predict.py    # Dataset build, train/test split, model comparison
├── backtest.py      # Long-short portfolio simulation + rebalance log
├── themes.csv       # Stock-to-theme mapping (ticker, theme, weight)
├── theme_agent.py   # Gemini agent: score themes using ETF context
├── env_loader.py    # Load GEMINI_API_KEY from .env
├── .env.example     # API key template (copy to .env)
├── theme_scores/    # Saved daily theme score JSON files
├── requirements.txt
└── README.md
```

---

## Methodology

### Universe & returns

- **Universe**: 50 liquid large-cap names (S&P 500 subset), configurable in `config.py`
- **Benchmark**: `SPY`
- **Excess return**: `y_excess(t,i) = r_stock(t,i) − r_SPY(t)`
- **Sample period**: 2016-01-01 → 2026-06-16 (default)

### Style factors (4)

Daily rolling exposures in `features.py`, z-scored cross-sectionally each day:

| Factor | Construction | Notes |
|--------|--------------|-------|
| **Size** | `ln(price)` | Price proxy for market cap in panel mode |
| **Value** | `book_per_share / price` | B/P proxy from current P/B × daily price |
| **Momentum** | Cumulative return over 63d, skip last 5d | Classic 12-1 style window |
| **Volatility** | 20d rolling std of daily returns | Low-vol anomaly exposure |

`barra.py` uses point-in-time `yfinance` snapshots for Size/Value (`ln(marketCap)`, `1/PB`) — useful for understanding WLS mechanics, not for production alpha.

### WLS factor return estimation

On a chosen date `t`, solve the cross-sectional regression:

```
β = (Xᵀ W X)⁻¹ Xᵀ W y
```

- `X ∈ ℝ^{N×M}`: standardized factor exposures
- `y ∈ ℝ^N`: excess returns
- `W = diag(ln(marketCap))`: capitalization weights (simplified; institutional Barra models typically use √cap)

`barra_panel.py` aligns `X` and `y` to the same ticker set before regression — some names drop out when Value data is missing.

### ML alpha model

**Features**: z-scored factor panel at date `t`  
**Label**: arithmetic sum of forward excess returns over `FORWARD_DAYS` (default 5):

```
target_{N}d(t,i) = Σ_{k=1}^{N} y_excess(t+k, i)
```

**Train/test split**: chronological, 80/20 by trading date — no random shuffle (avoids look-ahead bias).

**Models compared**:

| Model | Purpose |
|-------|---------|
| Baseline (ŷ = 0) | Null ranking benchmark |
| Momentum OLS | Single-factor linear baseline |
| Ridge (4 factors) | Regularized multi-factor linear model |
| RandomForest (4 factors) | Non-linear ensemble (`max_depth=3`) |

### Evaluation framework

**Primary metric — Mean Rank IC** (Spearman correlation between predictions and realized forward returns, averaged across test-set dates). Rank IC is the standard cross-sectional alpha metric in systematic equity research because it measures *ordering* ability, not level forecasting accuracy.

**Secondary metrics**:

| Metric | Definition |
|--------|------------|
| Mean IC | Daily Pearson(pred, actual), time-averaged |
| MSE | Global mean squared error on test set |
| Ann Return | Annualized long-short return |
| Sharpe | Annualized Sharpe on daily LS returns |
| Max Drawdown | Peak-to-trough on cumulative LS equity |
| Hit Rate | Fraction of days with positive LS PnL |

### Long-short backtest (`backtest.py`)

Portfolio construction on the **test set only**:

1. At each rebalance date, rank stocks by model prediction `pred`
2. **Long** top `TOP_PCT` (20%), **short** bottom 20%, equal-weighted
3. Hold until next rebalance; daily PnL = mean(long `ret_1d`) − mean(short `ret_1d`) − costs
4. Default: monthly rebalance (first trading day of month); `COST_BPS = 0`

**Design note**: ML label uses a 5-day forward horizon; backtest PnL uses next-day excess return (`ret_1d`) with monthly rebalancing. This is a deliberate simplification — in production, label horizon, holding period, and rebalance frequency should be aligned.

`long_short_backtest()` returns `(daily_ls, holdings_df)` with per-rebalance position logs (`date`, `side`, `ticker`, `pred`).

---

## Results (default config, out-of-sample test set)

`START_DATE = 2016-01-01`, 50 tickers, monthly rebalance, zero transaction costs:

```
--- ML dataset ---
Rows: 122784 | Features: ['Size', 'Value', 'Momentum', 'Volatility']
Label:  forward 5-day excess return
Train:  98208 rows (through 2024-05-21)
Test:   24576 rows (from 2024-05-22)

--- Model comparison (test set) ---
                          mean_ic  mean_rank_ic  ann_return  sharpe  max_drawdown  hit_rate
Ridge (4 factors)          0.0730        0.0546      0.5787  1.9521       -0.2023    0.5586
RandomForest (4 factors)   0.0197        0.0373      0.3174  1.4142       -0.1906    0.5430
Momentum only (OLS)        0.0246        0.0216      0.2193  0.8045       -0.2722    0.5391
Baseline (predict 0)          NaN           NaN     -0.2166 -1.8575       -0.4113    0.4531
```

**How to read this**:

- Ridge achieves **Rank IC ≈ 0.055** — modest but directionally meaningful for a 50-stock universe with free data and no neutralization.
- Linear multi-factor model beats single-factor Momentum and non-linear RandomForest on Rank IC — suggests signal is largely linear in these features, or RF is underfit/overfit relative to sample size.
- Backtest returns are **gross of costs** on a small, survivorship-biased universe over a short OOS window — treat PnL as illustrative, not investable.

---

## Setup & Run

**Requirements**: Python 3.11+, see `requirements.txt` (`yfinance`, `pandas`, `numpy`, `scikit-learn`, `google-genai`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Gemini theme agent (copy template, add your key)
cp .env.example .env
# Edit .env → GEMINI_API_KEY=...

python barra.py          # WLS demo (~seconds)
python barra_panel.py    # Factor panel + WLS
python ml_predict.py     # Full pipeline (~minutes on first run)
python theme_agent.py    # Score themes via Gemini
python theme_agent.py --mock   # ETF-momentum proxy, no API key
```

First run with 50 tickers is slow: bulk price download plus per-ticker `yfinance` info calls for the Value factor.

**Gemini API key**: Get one at [Google AI Studio](https://aistudio.google.com/apikey). Store it in `.env` (gitignored). Use the API model ID in `config.py` (e.g. `gemini-2.5-flash-lite`), not the display name from the UI.

---

## Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKERS` | 50 large caps | Stock universe |
| `BENCHMARK` | `SPY` | Excess return benchmark |
| `START_DATE` / `END_DATE` | 2016 → 2026 | Data window |
| `LOOKBACK_MOM` / `SKIP_RECENT` | 63 / 5 | Momentum window |
| `LOOKBACK_VOL` | 20 | Volatility window |
| `FORWARD_DAYS` | 5 | ML label horizon |
| `TRAIN_RATIO` | 0.8 | Chronological train fraction |
| `TOP_PCT` | 0.2 | Long/short leg size |
| `COST_BPS` | 0 | One-way cost on rebalance days |
| `REBALANCE_FREQ` | `monthly` | `monthly` or `daily` |
| `THEMES_FILE` | `themes.csv` | Stock-theme mapping |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model ID for theme scoring |
| `THEME_SCORES_DIR` | `theme_scores` | Output directory for daily scores |
| `ENV_FILE` | `.env` | Local API key file (gitignored) |

---

## AI Theme Agent

**Status: implemented** (`theme_agent.py`, `themes.csv`).

### Concept

Instead of treating ML predictions directly as the AI factor, this layer uses a **thematic investing** approach:

1. Each stock maps to one or more **themes** with weights (`themes.csv`).
2. A **Gemini agent** scores each theme's near-term attractiveness on a scale of **[-1, +1]**, grounded in recent ETF proxy returns.
3. Stock-level AI exposure (planned in `ai_factor.py`):

```
AI_Exposure(t, i) = Σ_theme  weight(i, theme) × ThemeScore(t, theme)
```

After cross-sectional z-scoring, this becomes the **5th column in X** for WLS.

### Themes & mapping

10 themes across the 50-stock universe: `AI`, `Cloud`, `Consumer`, `Energy`, `Financials`, `Healthcare`, `Industrials`, `Media`, `Semiconductors`, `Software`.

Example from `themes.csv`:

```csv
ticker,theme,weight
NVDA,AI,0.55
NVDA,Semiconductors,0.45
TSLA,Industrials,0.45
TSLA,Energy,0.3
TSLA,AI,0.25
```

### How the agent works

```mermaid
flowchart LR
    CSV[themes.csv] --> THEMES[10 themes]
    ETF[Theme ETF proxies] --> CTX[1m / 3m returns]
    CTX --> GEM[Gemini agent]
    THEMES --> GEM
    GEM --> JSON[theme_scores/YYYY-MM-DD.json]
```

For each theme, `theme_agent.py` fetches a sector ETF proxy (e.g. AI → `BOTZ`, Semiconductors → `SMH`) and passes recent momentum to Gemini along with a structured prompt. The agent returns JSON scores, saved to `theme_scores/`.

| Theme | ETF proxy |
|-------|-----------|
| AI | BOTZ |
| Cloud | WCLD |
| Semiconductors | SMH |
| Healthcare | XLV |
| Financials | XLF |
| Consumer | XLY |
| Energy | XLE |
| Media | XLC |
| Industrials | XLI |
| Software | IGV |

### CLI

```bash
python theme_agent.py                  # Gemini (reads .env)
python theme_agent.py --date 2026-06-19
python theme_agent.py --mock           # ETF z-score proxy, no API key
python theme_agent.py --no-save        # print only
```

Example output (`theme_scores/2026-06-19.json`):

```json
{
  "as_of": "2026-06-19",
  "source": "gemini",
  "scores": {
    "Semiconductors": 0.9,
    "Financials": 0.7,
    "Energy": -0.8
  }
}
```

### Limitations

- Theme mapping is **static** (does not evolve as business models change).
- Gemini scores are **point-in-time** — serious backtesting needs historical score snapshots or a reproducible proxy (`--mock`).
- Theme scores may correlate with **Momentum**; orthogonality should be checked before WLS embedding.

---

## Planned: AI Factor

**Status: partially implemented.** Theme scoring is live (`theme_agent.py`); stock-level exposure panel and 5-factor WLS integration are next.

### Concept

Embed the AI thematic signal into the multi-factor risk/return framework:

```
AI_Exposure(t, i) = z-score_cross_section( Σ_theme weight(i,theme) × ThemeScore(t,theme) )
```

This exposure joins the four style factors as a **5th column in X**, and WLS estimates its daily factor premium:

```
r_i − r_SPY ≈ β_Size·X_Size + … + β_AI·X_AI + ε_i
```

### Intended workflow

```mermaid
flowchart LR
    subgraph done [Implemented]
        CSV[themes.csv] --> AGENT[Gemini theme agent]
        AGENT --> SCORES[theme_scores JSON]
    end

    subgraph next [Next — ai_factor.py]
        SCORES --> EXP[Stock AI exposure panel]
        CSV --> EXP
        EXP --> ZS[Cross-sectional z-score]
        ZS --> X5["X panel: 5 factors"]
        F4["4 style factors"] --> X5
        X5 --> WLS2["WLS → β_AI and other premia"]
        WLS2 --> CMP["Compare: 4-factor vs 5-factor IC / R²"]
    end
```

### Remaining implementation

| Step | Module | Change |
|------|--------|--------|
| Map theme scores → stock exposure | `ai_factor.py` (new) | `build_ai_exposure(scores, theme_map)` |
| Merge into factor panel | `features.py` | Join AI column in `build_factor_panel()` |
| Extend factor universe | `config.py` | `FACTOR_NAMES` → add `"AI"` |
| WLS with 5 factors | `barra_panel.py` | Regress on extended `X`; report `AI_Return` |
| Evaluation | `ml_predict.py` | Incremental Rank IC, factor correlation with Momentum |

### Research questions the AI factor would answer

- Does `β_AI` have a stable positive premium out-of-sample?
- Is the AI theme signal orthogonal to classical factors, or does it overlap with Momentum?
- Does adding AI to WLS improve cross-sectional R² beyond the 4-factor model?
- Does Gemini-based theme scoring add information beyond raw ETF momentum (`--mock` baseline)?

### Why full integration isn't done yet

- Need **historical theme score panels** for walk-forward WLS (daily Gemini calls are expensive; mock proxy or cached JSON for backtest).
- Factor correlation and turnover need to be checked before treating AI as a style exposure.

---

## Known Limitations (and what I'd fix next)

Demonstrates research judgment — important for quant interviews:

| Issue | Impact | Production fix |
|-------|--------|----------------|
| Survivorship-biased 50-stock snapshot | Inflates backtest | Point-in-time index membership (CRSP/Compustat) |
| Value from static P/B, not quarterly fundamentals | Look-ahead / stale B/P | Forward-filled quarterly book equity |
| Size = `ln(price)` in panel | Imperfect cap proxy | Shares outstanding × price |
| Single train/test split | Overfit risk to one regime | Walk-forward / purged cross-validation |
| No sector neutralization | Factor crowding in tech, etc. | Industry constraints or orthogonalization |
| Label horizon ≠ backtest holding period | PnL not tied to prediction horizon | Align N-day label with N-day hold or overlap-adjusted IC |
| Zero transaction costs | Overstates net alpha | Realistic bps + market impact model |
| No specific risk / covariance | Can't size positions optimally | Barra USE4-style risk model |
| Static theme mapping | Misclassifies evolving businesses | Point-in-time sector/thematic tags |
| Gemini scores not historical | Can't backtest AI factor rigorously | Cached daily scores or ETF proxy baseline |

---

## Research Extensions

Beyond the [AI factor integration](#planned-ai-factor):

1. **`ai_factor.py`** — Map theme scores to stock exposures; merge into WLS / ML pipeline
2. **Walk-forward validation** — Rolling train/refit windows with decay-weighted samples
3. **Neutralized alpha** — Regress predictions on industry dummies; trade residual signal
4. **Alternative labels** — Rank-transformed returns, volatility-scaled targets, quintile classification
5. **Robustness** — Subperiod analysis, turnover-adjusted Sharpe, deflated Sharpe ratio

---

## Interview Talking Points

Questions this repo is designed to support:

- **Why Rank IC over MSE?** Cross-sectional strategies care about relative ordering, not absolute return levels. A model with low MSE but random ranks has no alpha.
- **Why chronological split?** Random splits leak future information into training — a common failure mode in quant ML.
- **Why does Ridge beat RandomForest here?** Limited non-linearity in linear style factors, small feature space, and regularization helping with multicollinearity among Size/Value/Momentum.
- **Why are backtest returns so high?** Small universe, no costs, survivorship bias, and a favorable OOS window — I report them with caveats, not as live estimates.
- **How would you productionize this?** Point-in-time data, walk-forward, neutralization, risk model integration, and realistic execution simulation.
- **What's the AI factor?** Gemini scores investment themes; stock exposure is a weighted sum of theme scores from `themes.csv`, z-scored and embedded as a 5th WLS factor. Theme agent is built; `ai_factor.py` integration is next.
- **How do you backtest LLM-based signals?** Point-in-time snapshots in `theme_scores/`, or a reproducible ETF-momentum baseline (`--mock`) for comparison against Gemini.

---

## License

Personal / educational use.

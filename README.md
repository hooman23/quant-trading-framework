# quant-trading-framework
Systematic trading framework backtested on S&amp;P 500 — 4 models, net of commission &amp; slippage
"""
================================================================================
QUANTITATIVE TRADING FRAMEWORK — S&P 500
================================================================================
Models:
  1. Mean Reversion        — Bollinger Bands + Z-score entry/exit
  2. Volatility Regime     — VIX-proxy regime switching (low/high vol)
  3. Distribution Shift    — KL-divergence / CUSUM change-point detection
  4. Trend Following       — Dual EMA crossover with ATR trailing stop

Benchmark: S&P 500 Buy-and-Hold
Metrics:   Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor,
           Calmar, Win Rate, Avg Win/Loss, Commission + Slippage adjusted
================================================================================
To use REAL data, replace generate_sp500_data() with:
    import yfinance as yf
    df = yf.download("^GSPC", start="2010-01-01", end="2024-01-01")
    df = df[["Open","High","Low","Close","Volume"]].dropna()
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
COMMISSION  = 0.001   # 0.10% per side
SLIPPAGE    = 0.0005  # 0.05% per side
INITIAL_CAP = 100_000
START_DATE  = "2010-01-01"
N_DAYS      = 3500    # ~14 years


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION  (realistic S&P 500 GBM + GARCH-like vol clustering)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sp500_data(n_days: int = N_DAYS, seed: int = 42) -> pd.DataFrame:
    """
    Simulate S&P 500-like daily OHLCV data using:
    - GARCH(1,1) volatility clustering
    - Fat-tailed returns (Student-t, df=5)
    - Realistic drift (~7% annual)
    - Intraday OHLC construction
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START_DATE, periods=n_days)

    # GARCH(1,1) parameters (fitted to S&P 500)
    omega, alpha, beta = 0.000002, 0.09, 0.90
    mu   = 0.00025      # ~6.5% annual drift
    df_t = 5            # Student-t degrees of freedom

    price = 1100.0
    h     = omega / (1 - alpha - beta)   # unconditional variance
    closes, highs, lows, opens = [], [], [], []
    vols = []

    for _ in range(n_days):
        h   = omega + alpha * (rng.standard_t(df_t) * np.sqrt(h))**2 + beta * h
        ret = mu + np.sqrt(h) * rng.standard_t(df_t)
        ret = np.clip(ret, -0.12, 0.12)

        o = price
        c = price * np.exp(ret)
        intra_range = abs(ret) + rng.exponential(0.005)
        h_p = max(o, c) * (1 + rng.uniform(0, intra_range))
        l_p = min(o, c) * (1 - rng.uniform(0, intra_range))

        opens.append(o); closes.append(c)
        highs.append(h_p); lows.append(l_p)
        vols.append(abs(ret))
        price = c

    df = pd.DataFrame({
        "Open":  opens, "High": highs,
        "Low":   lows,  "Close": closes,
        "Volume": rng.integers(1_000_000, 5_000_000, n_days)
    }, index=dates)

    # Synthetic "VIX" proxy: 21-day realised vol * sqrt(252) * 100
    df["VIX"] = pd.Series(vols, index=dates).rolling(21).std() * np.sqrt(252) * 100
    df["VIX"] = df["VIX"].bfill()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UTILITY — TRANSACTION COST
# ═══════════════════════════════════════════════════════════════════════════════

def apply_costs(price: float, direction: str) -> float:
    """Return execution price after commission + slippage."""
    cost = COMMISSION + SLIPPAGE
    if direction == "buy":
        return price * (1 + cost)
    return price * (1 - cost)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BACKTESTER  (event-driven, daily bar)
# ═══════════════════════════════════════════════════════════════════════════════

def backtest(signals: pd.Series, prices: pd.DataFrame,
             label: str = "Strategy") -> dict:
    """
    signals: pd.Series aligned to prices.index
             +1 = long, -1 = short/flat, 0 = flat
    Returns dict of equity curve + metrics.
    """
    close   = prices["Close"]
    equity  = [INITIAL_CAP]
    cash    = INITIAL_CAP
    pos     = 0          # shares held
    entry_p = 0.0
    trades  = []

    for i in range(1, len(close)):
        sig_prev = signals.iloc[i - 1]
        price    = close.iloc[i]

        # ── ENTRY ──────────────────────────────
        if sig_prev == 1 and pos == 0:
            exec_p  = apply_costs(price, "buy")
            pos     = cash / exec_p
            cash    = 0.0
            entry_p = exec_p

        # ── EXIT ───────────────────────────────
        elif sig_prev != 1 and pos > 0:
            exec_p  = apply_costs(price, "sell")
            pnl     = (exec_p - entry_p) / entry_p
            trades.append(pnl)
            cash    = pos * exec_p
            pos     = 0.0

        nav = cash + pos * price
        equity.append(nav)

    # close any open position at end
    if pos > 0:
        exec_p = apply_costs(close.iloc[-1], "sell")
        pnl    = (exec_p - entry_p) / entry_p
        trades.append(pnl)
        equity[-1] = pos * exec_p

    eq = pd.Series(equity, index=prices.index)
    return _metrics(eq, trades, label)


def _metrics(equity: pd.Series, trades: list, label: str) -> dict:
    rets   = equity.pct_change().dropna()
    years  = len(equity) / 252

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr      = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    sharpe    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0

    down      = rets[rets < 0]
    sortino   = (rets.mean() / down.std()) * np.sqrt(252) if len(down) > 0 else 0

    roll_max  = equity.cummax()
    dd        = (equity - roll_max) / roll_max
    max_dd    = dd.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0

    wins      = [t for t in trades if t > 0]
    losses    = [t for t in trades if t <= 0]
    win_rate  = len(wins) / len(trades) if trades else 0
    avg_win   = np.mean(wins)  if wins   else 0
    avg_loss  = np.mean(losses) if losses else 0
    gross_p   = sum(wins)
    gross_l   = abs(sum(losses))
    pf        = gross_p / gross_l if gross_l > 0 else np.inf

    return {
        "label":     label,
        "equity":    equity,
        "drawdown":  dd,
        "trades":    trades,
        "metrics": {
            "Total Return (%)":   round(total_ret * 100, 2),
            "CAGR (%)":           round(cagr * 100, 2),
            "Sharpe Ratio":       round(sharpe, 3),
            "Sortino Ratio":      round(sortino, 3),
            "Max Drawdown (%)":   round(max_dd * 100, 2),
            "Calmar Ratio":       round(calmar, 3),
            "Profit Factor":      round(pf, 3),
            "Win Rate (%)":       round(win_rate * 100, 2),
            "Avg Win (%)":        round(avg_win * 100, 3),
            "Avg Loss (%)":       round(avg_loss * 100, 3),
            "# Trades":           len(trades),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4a. MODEL 1 — MEAN REVERSION (Bollinger + Z-Score)
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_mean_reversion(df: pd.DataFrame,
                             window: int = 20,
                             z_entry: float = -1.5,
                             z_exit:  float =  0.5) -> pd.Series:
    """
    Logic:
    - Compute rolling z-score of Close vs. its 20-day mean/std
    - Enter long when z < -1.5 (price below lower band → mean reversion expected)
    - Exit long when z > +0.5 (price reverts to or above mean)
    - Uses Bollinger Band upper/lower for visual confirmation
    """
    close   = df["Close"]
    ma      = close.rolling(window).mean()
    std     = close.rolling(window).std()
    z_score = (close - ma) / std

    # Lower / upper bands (2σ)
    df["BB_upper"] = ma + 2 * std
    df["BB_lower"] = ma - 2 * std
    df["BB_mid"]   = ma
    df["Z_score"]  = z_score

    signal = pd.Series(0, index=df.index)
    in_trade = False

    for i in range(window, len(df)):
        z = z_score.iloc[i]
        if not in_trade and z < z_entry:
            in_trade = True
        elif in_trade and z > z_exit:
            in_trade = False
        signal.iloc[i] = 1 if in_trade else 0

    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. MODEL 2 — VOLATILITY REGIME (VIX-proxy switching)
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_volatility_regime(df: pd.DataFrame,
                                vix_low: float  = 18.0,
                                vix_high: float = 28.0,
                                trend_window: int = 50) -> pd.Series:
    """
    Logic:
    - Low vol regime  (VIX < 18):  trend-following mode → long if price > 50d MA
    - High vol regime (VIX > 28):  mean-reversion mode  → contrarian entries
    - Mid vol:                      hold / neutral
    Regime is smoothed with a 5-day EMA to avoid whipsaws.
    """
    close     = df["Close"]
    vix       = df["VIX"].ewm(span=5).mean()
    ma50      = close.rolling(trend_window).mean()

    signal = pd.Series(0, index=df.index)

    for i in range(trend_window, len(df)):
        v  = vix.iloc[i]
        c  = close.iloc[i]
        m  = ma50.iloc[i]

        if v < vix_low:          # Low vol → trend follow
            signal.iloc[i] = 1 if c > m else 0
        elif v > vix_high:       # High vol → mean-revert (buy dips)
            z = (c - ma50.iloc[i]) / close.rolling(20).std().iloc[i]
            signal.iloc[i] = 1 if z < -1.0 else 0
        else:                    # Mid vol → flat
            signal.iloc[i] = signal.iloc[i - 1]   # hold last position

    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 4c. MODEL 3 — DISTRIBUTION SHIFT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """Symmetric KL divergence (Jensen-Shannon) between two return distributions."""
    p_hist, edges = np.histogram(p, bins=bins, density=True)
    q_hist, _     = np.histogram(q, bins=edges,  density=True)
    p_hist = np.clip(p_hist, 1e-10, None)
    q_hist = np.clip(q_hist, 1e-10, None)
    p_hist /= p_hist.sum(); q_hist /= q_hist.sum()
    m = 0.5 * (p_hist + q_hist)
    return 0.5 * (np.sum(p_hist * np.log(p_hist / m)) +
                  np.sum(q_hist * np.log(q_hist / m)))


def strategy_distribution_shift(df: pd.DataFrame,
                                 reference_window: int = 126,  # ~6 months
                                 detection_window: int =  21,  # ~1 month
                                 kl_threshold: float   =  0.08,
                                 cusum_threshold: float = 5.0) -> pd.Series:
    """
    Logic:
    - Reference distribution: rolling 126-day return window
    - Detection window: most recent 21 days
    - Compute KL divergence; if KL > threshold → distribution shift detected
    - CUSUM on log-returns detects persistent mean shift
    - AVOID market when shift detected (stay out), re-enter when conditions normalise
    - Acts as a risk-off overlay on a base buy-signal
    """
    rets    = df["Close"].pct_change()
    signal  = pd.Series(0, index=df.index)
    df["KL_div"] = np.nan
    df["CUSUM"]  = 0.0

    cusum_pos = 0.0
    cusum_neg = 0.0
    mu_target = rets.mean()

    for i in range(reference_window + detection_window, len(df)):
        ref   = rets.iloc[i - reference_window - detection_window : i - detection_window].dropna().values
        det   = rets.iloc[i - detection_window : i].dropna().values

        if len(ref) < 10 or len(det) < 5:
            continue

        kl    = _kl_divergence(ref, det)
        df.at[df.index[i], "KL_div"] = kl

        # CUSUM
        r = rets.iloc[i]
        slack = 0.5 * rets.std()
        cusum_pos = max(0, cusum_pos + (r - mu_target) - slack)
        cusum_neg = max(0, cusum_neg - (r - mu_target) - slack)
        df.at[df.index[i], "CUSUM"] = cusum_pos - cusum_neg

        shift_detected = (kl > kl_threshold) or \
                         (cusum_pos > cusum_threshold) or \
                         (cusum_neg > cusum_threshold)

        # Base condition: simple momentum (5d > 20d MA)
        ma5  = df["Close"].iloc[max(0,i-5):i].mean()
        ma20 = df["Close"].iloc[max(0,i-20):i].mean()
        base_long = ma5 > ma20

        # Risk-off when shift detected
        signal.iloc[i] = 1 if (base_long and not shift_detected) else 0

        # Reset CUSUM after detection + exit
        if shift_detected:
            cusum_pos = 0; cusum_neg = 0

    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 4d. MODEL 4 — TREND FOLLOWING (Dual EMA + ATR Trailing Stop)
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_trend_following(df: pd.DataFrame,
                              fast: int = 20, slow: int = 50,
                              atr_mult: float = 2.5,
                              atr_window: int = 14) -> pd.Series:
    """
    Logic:
    - Fast EMA crosses above slow EMA → enter long
    - Fast EMA crosses below slow EMA → exit
    - ATR trailing stop provides hard exit on adverse moves
    - Trend filter: only trade when 200d MA is rising (uptrend)
    """
    close = df["Close"]
    ema_f = close.ewm(span=fast,  adjust=False).mean()
    ema_s = close.ewm(span=slow,  adjust=False).mean()
    ma200 = close.rolling(200).mean()

    # ATR
    tr1  = df["High"] - df["Low"]
    tr2  = (df["High"] - df["Close"].shift()).abs()
    tr3  = (df["Low"]  - df["Close"].shift()).abs()
    atr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(atr_window).mean()

    signal     = pd.Series(0, index=df.index)
    trail_stop = 0.0
    in_trade   = False

    for i in range(200, len(df)):
        c  = close.iloc[i]
        ef = ema_f.iloc[i]; es = ema_s.iloc[i]
        trend_up = close.iloc[i] > ma200.iloc[i] and \
                   ma200.iloc[i] > ma200.iloc[i - 10]

        if not in_trade:
            cross_up = (ema_f.iloc[i] > ema_s.iloc[i]) and \
                       (ema_f.iloc[i-1] <= ema_s.iloc[i-1])
            if cross_up and trend_up:
                in_trade   = True
                trail_stop = c - atr_mult * atr.iloc[i]
        else:
            trail_stop = max(trail_stop, c - atr_mult * atr.iloc[i])
            cross_dn   = ef < es
            if cross_dn or c < trail_stop:
                in_trade   = False
                trail_stop = 0.0

        signal.iloc[i] = 1 if in_trade else 0

    df["EMA_fast"] = ema_f; df["EMA_slow"] = ema_s
    df["MA200"]    = ma200; df["ATR"]      = atr
    return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BENCHMARK — BUY & HOLD
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_buyhold(df: pd.DataFrame) -> dict:
    exec_buy  = apply_costs(df["Close"].iloc[0],  "buy")
    exec_sell = apply_costs(df["Close"].iloc[-1], "sell")
    shares    = INITIAL_CAP / exec_buy
    equity    = (df["Close"] / df["Close"].iloc[0]) * (shares * exec_buy)
    ret       = (exec_sell - exec_buy) / exec_buy
    return _metrics(equity, [ret], "S&P 500 Buy-and-Hold")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RUN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(df: pd.DataFrame) -> dict:
    results = {}
    results["benchmark"] = benchmark_buyhold(df)

    sig_mr  = strategy_mean_reversion(df.copy())
    sig_vr  = strategy_volatility_regime(df.copy())
    sig_ds  = strategy_distribution_shift(df.copy())
    sig_tf  = strategy_trend_following(df.copy())

    results["mean_rev"]   = backtest(sig_mr, df, "Mean Reversion")
    results["vol_regime"] = backtest(sig_vr, df, "Volatility Regime")
    results["dist_shift"] = backtest(sig_ds, df, "Distribution Shift")
    results["trend"]      = backtest(sig_tf, df, "Trend Following")

    # Attach signals for inspection
    results["signals"] = {
        "mean_rev":   sig_mr,
        "vol_regime": sig_vr,
        "dist_shift": sig_ds,
        "trend":      sig_tf,
    }

    # Store indicator columns for plots
    results["df"] = df
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TEARSHEET — MATPLOTLIB
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "benchmark":  "#94a3b8",
    "mean_rev":   "#3b82f6",
    "vol_regime": "#10b981",
    "dist_shift": "#f59e0b",
    "trend":      "#ef4444",
}

def plot_tearsheet(results: dict, save_path: str = "/home/claude/tearsheet.png"):
    df  = results["df"]
    bm  = results["benchmark"]

    fig = plt.figure(figsize=(22, 28), facecolor="#0f172a")
    fig.patch.set_facecolor("#0f172a")
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            hspace=0.55, wspace=0.35,
                            top=0.93, bottom=0.04,
                            left=0.07, right=0.97)

    txt_kw   = dict(color="#e2e8f0", fontfamily="monospace")
    title_kw = dict(color="#f8fafc", fontweight="bold", fontsize=11)
    ax_bg    = "#1e293b"
    grid_c   = "#334155"

    def style_ax(ax):
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.spines[:].set_color(grid_c)
        ax.yaxis.label.set_color("#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.grid(True, color=grid_c, linewidth=0.4, linestyle="--", alpha=0.5)

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965, "QUANTITATIVE TRADING SYSTEM — S&P 500",
             ha="center", fontsize=18, fontweight="bold",
             color="#f8fafc", fontfamily="monospace")
    fig.text(0.5, 0.950, "Multi-Model Framework  |  Commission 0.10%  |  Slippage 0.05%  |  ~14 Year Backtest",
             ha="center", fontsize=10, color="#64748b", fontfamily="monospace")

    # ── 1. Equity Curves ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)
    ax1.set_title("Equity Curves (log scale)", **title_kw)

    for key in ["benchmark", "mean_rev", "vol_regime", "dist_shift", "trend"]:
        r = results[key]
        ax1.semilogy(r["equity"].index, r["equity"],
                     color=COLORS[key], linewidth=1.6, label=r["label"],
                     alpha=0.9)

    ax1.legend(loc="upper left", fontsize=9, framealpha=0.2,
               facecolor="#1e293b", edgecolor="#475569",
               labelcolor="#e2e8f0")
    ax1.set_ylabel("Portfolio Value ($)")

    # ── 2. Drawdown ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    style_ax(ax2)
    ax2.set_title("Drawdown", **title_kw)

    for key in ["benchmark", "mean_rev", "vol_regime", "dist_shift", "trend"]:
        r = results[key]
        ax2.fill_between(r["drawdown"].index,
                         r["drawdown"] * 100, 0,
                         color=COLORS[key], alpha=0.35, label=r["label"])
        ax2.plot(r["drawdown"].index, r["drawdown"] * 100,
                 color=COLORS[key], linewidth=0.8)

    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.2,
               facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0")

    # ── 3. Rolling Sharpe (63-day) ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    style_ax(ax3)
    ax3.set_title("Rolling 63-Day Sharpe", **title_kw)

    for key in ["mean_rev", "vol_regime", "dist_shift", "trend"]:
        r   = results[key]
        ret = r["equity"].pct_change().dropna()
        rs  = ret.rolling(63).mean() / ret.rolling(63).std() * np.sqrt(252)
        ax3.plot(rs.index, rs, color=COLORS[key], linewidth=1.2,
                 label=results[key]["label"], alpha=0.85)

    ax3.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("Sharpe")
    ax3.legend(fontsize=8, framealpha=0.2, facecolor="#1e293b",
               edgecolor="#475569", labelcolor="#e2e8f0")

    # ── 4. Return Distribution ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    style_ax(ax4)
    ax4.set_title("Daily Return Distribution", **title_kw)

    for key in ["benchmark", "mean_rev", "vol_regime", "dist_shift", "trend"]:
        r   = results[key]
        ret = r["equity"].pct_change().dropna() * 100
        ax4.hist(ret, bins=80, alpha=0.45, color=COLORS[key],
                 label=results[key]["label"], density=True)

    ax4.set_xlabel("Daily Return (%)")
    ax4.set_ylabel("Density")
    ax4.set_xlim(-4, 4)
    ax4.legend(fontsize=8, framealpha=0.2, facecolor="#1e293b",
               edgecolor="#475569", labelcolor="#e2e8f0")

    # ── 5. Win Rate Bar ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    style_ax(ax5)
    ax5.set_title("Win Rate by Strategy", **title_kw)

    names = []; wr = []; pf = []
    for key in ["mean_rev", "vol_regime", "dist_shift", "trend"]:
        r = results[key]
        names.append(r["label"].replace(" ", "\n"))
        wr.append(r["metrics"]["Win Rate (%)"])
        pf.append(min(r["metrics"]["Profit Factor"], 5))

    x    = np.arange(len(names))
    bars = ax5.bar(x, wr, color=[COLORS[k] for k in
           ["mean_rev","vol_regime","dist_shift","trend"]],
           alpha=0.85, width=0.5, zorder=3)
    ax5.set_xticks(x); ax5.set_xticklabels(names, fontsize=8, color="#94a3b8")
    ax5.set_ylabel("Win Rate (%)")
    ax5.axhline(50, color="#f59e0b", linewidth=1, linestyle="--", alpha=0.6)
    for bar, v in zip(bars, wr):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=9, **txt_kw)

    # ── 6. Profit Factor Bar ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    style_ax(ax6)
    ax6.set_title("Profit Factor (capped at 5)", **title_kw)

    bars2 = ax6.bar(x, pf, color=[COLORS[k] for k in
            ["mean_rev","vol_regime","dist_shift","trend"]],
            alpha=0.85, width=0.5, zorder=3)
    ax6.set_xticks(x); ax6.set_xticklabels(names, fontsize=8, color="#94a3b8")
    ax6.set_ylabel("Profit Factor")
    ax6.axhline(1.5, color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.9,
                label="PF = 1.5 target")
    ax6.axhline(1.0, color="#ef4444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax6.legend(fontsize=8, framealpha=0.2, facecolor="#1e293b",
               edgecolor="#475569", labelcolor="#e2e8f0")
    for bar, v in zip(bars2, pf):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{v:.2f}", ha="center", va="bottom",
                 fontsize=9, **txt_kw)

    # ── 7. Metrics Table ─────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    ax7.set_facecolor(ax_bg)
    ax7.axis("off")
    ax7.set_title("Performance Metrics Summary (net of commission + slippage)",
                  **title_kw, pad=10)

    keys_order = ["Total Return (%)", "CAGR (%)", "Sharpe Ratio",
                  "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio",
                  "Profit Factor", "Win Rate (%)", "# Trades"]

    col_labels  = ["Metric", "Buy & Hold",
                   "Mean Rev.", "Vol Regime", "Dist Shift", "Trend"]
    table_data  = []
    all_results = [results["benchmark"], results["mean_rev"],
                   results["vol_regime"], results["dist_shift"],
                   results["trend"]]

    for k in keys_order:
        row = [k]
        for r in all_results:
            row.append(str(r["metrics"].get(k, "—")))
        table_data.append(row)

    tbl = ax7.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(grid_c)
        if r == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="#93c5fd", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#e2e8f0")
        else:
            cell.set_facecolor("#172032")
            cell.set_text_props(color="#cbd5e1")

        # Highlight Profit Factor >= 1.5
        if r > 0 and c > 0 and keys_order[r-1] == "Profit Factor":
            try:
                val = float(tbl[r, c].get_text().get_text())
                if val >= 1.5:
                    cell.set_facecolor("#14532d")
                    cell.set_text_props(color="#4ade80", fontweight="bold")
            except:
                pass

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Tearsheet saved → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRINT CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    line = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  QUANTITATIVE STRATEGY RESULTS  |  Commission: {COMMISSION*100:.2f}%  |  Slippage: {SLIPPAGE*100:.2f}%")
    print(f"{'═'*72}\n")

    keys_order = ["Total Return (%)", "CAGR (%)", "Sharpe Ratio",
                  "Sortino Ratio", "Max Drawdown (%)", "Calmar Ratio",
                  "Profit Factor", "Win Rate (%)", "Avg Win (%)",
                  "Avg Loss (%)", "# Trades"]

    for key in ["benchmark", "mean_rev", "vol_regime", "dist_shift", "trend"]:
        r = results[key]
        print(f"  ► {r['label'].upper()}")
        print(f"  {line}")
        for k in keys_order:
            v   = r["metrics"].get(k, "—")
            tag = ""
            if k == "Profit Factor" and isinstance(v, float) and v >= 1.5:
                tag = "  ✅ BEATS TARGET"
            elif k == "Sharpe Ratio" and isinstance(v, float) and v > 1.0:
                tag = "  ✅"
            print(f"    {k:<25} {str(v):>12}{tag}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("[1/4] Generating S&P 500 simulation data...")
    df = generate_sp500_data()
    print(f"      {len(df)} trading days  |  {df.index[0].date()} → {df.index[-1].date()}")

    print("[2/4] Running strategies...")
    results = run_all(df)

    print("[3/4] Printing summary...")
    print_summary(results)

    print("[4/4] Generating tearsheet...")
    plot_tearsheet(results)
    print("\n[✓] All done.\n")

"""
NGX-FND Return-Sentiment Regression Analysis
=============================================
Tests whether LLM-extracted sentiment and guidance signals predict
post-disclosure abnormal returns on the Nigerian Exchange.

Models estimated:
  Model 1 -- Univariate: CAR = f(sentiment_score)
  Model 2 -- Guidance:   CAR = f(has_guidance, guidance_type)
  Model 3 -- Combined:   CAR = f(sentiment_score, guidance_score, controls)
  Model 4 -- Full:       CAR = f(all signals + sector FE + year FE)
  Model 5 -- Net sentiment: CAR = f(net_sentiment_index, guidance_adj)

Controls: log(market_cap), beta, sector FE, year FE, doc_type

Statistical tests:
  - OLS with Newey-West HAC standard errors (autocorrelation-robust)
  - Cross-sectional t-tests of mean CAR by sentiment group
  - Spearman rank correlations (non-parametric)
  - Bootstrap confidence intervals

Outputs:
  data/returns_analysis/regression_results.csv
  data/returns_analysis/group_tests.csv
  data/returns_analysis/paper_tables.tex
  data/figures/sentiment_vs_car.pdf
  data/figures/car_by_sentiment_group.pdf
  data/figures/event_window_plot.pdf

Usage:
  python return_regression.py
  python return_regression.py --car CAR[-1,+3]   # use specific window
  python return_regression.py --plots             # generate all figures
"""

import warnings
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hac

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
PRICES_DIR    = BASE_DIR / "data" / "prices"
ANALYSIS_DIR  = BASE_DIR / "data" / "returns_analysis"
FIGURES_DIR   = BASE_DIR / "data" / "figures"
LOG_DIR       = BASE_DIR / "logs"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# Plot styling
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "figure.dpi":      150,
})


# -- Data Loader ----------------------------------------------------------------

def load_analysis_data(car_col: str = "CAR[0,+3]") -> pd.DataFrame | None:
    path = PRICES_DIR / "abnormal_returns.csv"
    if not path.exists():
        log.error(f"Abnormal returns not found: {path}. Run abnormal_returns.py first.")
        return None

    df = pd.read_csv(path, parse_dates=["event_date"])

    # Validate required columns
    required = [car_col, "sentiment_score_mean", "ticker", "sector", "fiscal_year"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"Missing columns: {missing}. Some models will be skipped.")

    # Drop rows with no CAR or no sentiment
    df = df.dropna(subset=[car_col])
    log.info(f"Loaded {len(df)} events with valid {car_col}")

    # Feature engineering
    if "pct_positive" in df and "pct_negative" in df:
        df["net_sentiment"]   = df["pct_positive"] - df["pct_negative"]
    if "guidance_rate" in df:
        df["has_guidance_int"] = df["guidance_rate"].apply(lambda x: 1 if x > 0.3 else 0)
    if "beta" in df:
        df["beta_sq"]         = df["beta"] ** 2
    if "fiscal_year" in df:
        df["year_int"]        = pd.to_numeric(df["fiscal_year"], errors="coerce")
        df["post_fx_unif"]    = (df["year_int"] >= 2023).astype(int)   # FX unification dummy

    # Sector and year dummies (dtype=int to avoid bool issues in OLS)
    if "sector" in df:
        df = pd.get_dummies(df, columns=["sector"], prefix="sec", drop_first=True, dtype=int)
    if "fiscal_year" in df:
        df = pd.get_dummies(df, columns=["fiscal_year"], prefix="yr", drop_first=True, dtype=int)
    if "doc_type" in df:
        df = pd.get_dummies(df, columns=["doc_type"], prefix="dt", drop_first=True, dtype=int)

    return df


# -- OLS with HAC SEs -----------------------------------------------------------

def run_ols_hac(y: pd.Series, X: pd.DataFrame, label: str = "") -> dict:
    """
    OLS regression with Newey-West HAC standard errors.
    Returns dict with coefficients, t-stats, p-values, R2, adj-R2.
    """
    X_df    = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    X_const = sm.add_constant(X_df, has_constant="add")
    mask    = y.notna() & X_const.notna().all(axis=1)
    y_clean = y[mask]
    X_clean = X_const[mask]

    if len(y_clean) < 10:
        return {"error": f"Insufficient observations: {len(y_clean)}"}

    try:
        model      = sm.OLS(y_clean, X_clean).fit()
        model_hac  = model.get_robustcov_results(cov_type="HAC", maxlags=4)

        # HAC results return numpy arrays -- use exog_names for variable names
        var_names  = model.model.exog_names
        params     = np.asarray(model_hac.params)
        bse        = np.asarray(model_hac.bse)
        tvalues    = np.asarray(model_hac.tvalues)
        pvalues    = np.asarray(model_hac.pvalues)

        result = {
            "label":      label,
            "n_obs":      int(len(y_clean)),
            "r_squared":  round(float(model.rsquared), 4),
            "adj_r2":     round(float(model.rsquared_adj), 4),
            "f_stat":     round(float(model_hac.fvalue),   4) if hasattr(model_hac, "fvalue")  and model_hac.fvalue  is not None else np.nan,
            "f_pvalue":   round(float(model_hac.f_pvalue), 4) if hasattr(model_hac, "f_pvalue") and model_hac.f_pvalue is not None else np.nan,
            "aic":        round(float(model.aic), 2),
            "bic":        round(float(model.bic), 2),
            "coefficients": {},
        }

        for i, var in enumerate(var_names):
            result["coefficients"][var] = {
                "coef":    round(float(params[i]),   6),
                "std_err": round(float(bse[i]),      6),
                "t_stat":  round(float(tvalues[i]),  4),
                "p_value": round(float(pvalues[i]),  4),
                "sig":     _significance_stars(float(pvalues[i])),
            }

        return result
    except Exception as e:
        return {"error": str(e), "label": label}


def _significance_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# -- Regression Models ----------------------------------------------------------

def run_all_models(df: pd.DataFrame, car_col: str) -> list[dict]:
    """Run all five regression models."""
    results = []
    y = df[car_col]

    # -- Model 1: Univariate sentiment ------------------------------------------
    if "sentiment_score_mean" in df:
        X1 = df[["sentiment_score_mean"]]
        results.append(run_ols_hac(y, X1, label="M1: Univariate Sentiment"))

    # -- Model 2: Guidance signals ----------------------------------------------
    guid_cols = [c for c in ["guidance_rate", "guidance_score_mean"] if c in df]
    if guid_cols:
        X2 = df[guid_cols]
        results.append(run_ols_hac(y, X2, label="M2: Guidance Signals"))

    # -- Model 3: Combined sentiment + guidance --------------------------------
    base_cols = [c for c in ["sentiment_score_mean", "guidance_score_mean",
                              "guidance_rate", "weighted_score_mean"] if c in df]
    if len(base_cols) >= 2:
        X3 = df[base_cols]
        results.append(run_ols_hac(y, X3, label="M3: Sentiment + Guidance"))

    # -- Model 4: Combined + controls (beta, FX dummy) ------------------------
    control_cols = [c for c in ["beta", "post_fx_unif"] if c in df]
    X4_cols = base_cols + control_cols
    if len(X4_cols) >= 3:
        X4 = df[X4_cols].copy()
        results.append(run_ols_hac(y, X4, label="M4: Sentiment + Controls"))

    # -- Model 5: Full model with sector + year FE -----------------------------
    sec_cols  = [c for c in df.columns if c.startswith("sec_")]
    yr_cols   = [c for c in df.columns if c.startswith("yr_")]
    dt_cols   = [c for c in df.columns if c.startswith("dt_")]
    fe_cols   = sec_cols + yr_cols + dt_cols
    X5_cols   = X4_cols + fe_cols
    X5_cols   = [c for c in X5_cols if c in df.columns]
    if len(X5_cols) >= 3:
        X5 = df[X5_cols].copy()
        results.append(run_ols_hac(y, X5, label="M5: Full Model (Sector + Year FE)"))

    # -- Model 6: Net sentiment index ------------------------------------------
    if "net_sentiment" in df:
        X6_cols = ["net_sentiment"] + [c for c in ["guidance_adj_sentiment", "beta"] if c in df]
        X6 = df[X6_cols]
        results.append(run_ols_hac(y, X6, label="M6: Net Sentiment Index"))

    return results


# -- Group Tests ----------------------------------------------------------------

def run_group_tests(df: pd.DataFrame, car_col: str) -> pd.DataFrame:
    """
    Cross-sectional tests: mean CAR by sentiment group.
    Tests whether positive-sentiment disclosures have significantly higher
    post-event returns than negative-sentiment disclosures.
    """
    if "pred_sentiment" not in df.columns and "sentiment_score_mean" not in df.columns:
        return pd.DataFrame()

    rows = []
    y    = df[car_col].dropna()

    # Sentiment group assignment
    if "sentiment_score_mean" in df.columns:
        df = df.copy()
        df["sentiment_group"] = pd.cut(
            df["sentiment_score_mean"],
            bins=[-np.inf, -0.2, 0.2, np.inf],
            labels=["Negative", "Neutral", "Positive"]
        )
    else:
        return pd.DataFrame()

    groups = df.groupby("sentiment_group")[car_col].apply(list)

    for group_name, group_vals in groups.items():
        arr = np.array([v for v in group_vals if not np.isnan(v)])
        if len(arr) < 5:
            continue
        t_stat, p_val = stats.ttest_1samp(arr, 0)
        rows.append({
            "group":    group_name,
            "n":        len(arr),
            "mean_car": round(arr.mean(), 4),
            "std_car":  round(arr.std(), 4),
            "median_car":round(np.median(arr), 4),
            "t_stat":   round(t_stat, 4),
            "p_value":  round(p_val, 4),
            "sig":      _significance_stars(p_val),
            "pct_positive_car": round((arr > 0).mean(), 4),
        })

    # Positive vs negative t-test
    if "Positive" in groups.index and "Negative" in groups.index:
        pos = np.array([v for v in groups["Positive"] if not np.isnan(v)])
        neg = np.array([v for v in groups["Negative"] if not np.isnan(v)])
        if len(pos) >= 5 and len(neg) >= 5:
            t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False)
            rows.append({
                "group":    "Positive vs Negative (diff)",
                "n":        f"{len(pos)} vs {len(neg)}",
                "mean_car": round(pos.mean() - neg.mean(), 4),
                "std_car":  np.nan,
                "median_car":round(np.median(pos) - np.median(neg), 4),
                "t_stat":   round(t_stat, 4),
                "p_value":  round(p_val, 4),
                "sig":      _significance_stars(p_val),
                "pct_positive_car": np.nan,
            })

    # Spearman correlation
    if "sentiment_score_mean" in df.columns:
        valid = df[[car_col, "sentiment_score_mean"]].dropna()
        if len(valid) >= 10:
            rho, p_rho = stats.spearmanr(valid[car_col], valid["sentiment_score_mean"])
            rows.append({
                "group":    "Spearman rho (sentiment vs CAR)",
                "n":        len(valid),
                "mean_car": round(rho, 4),
                "std_car":  np.nan,
                "median_car":np.nan,
                "t_stat":   np.nan,
                "p_value":  round(p_rho, 4),
                "sig":      _significance_stars(p_rho),
                "pct_positive_car": np.nan,
            })

    return pd.DataFrame(rows)


# -- Robustness Checks ----------------------------------------------------------

def run_placebo_test(df: pd.DataFrame, car_col: str,
                     n_permutations: int = 1000,
                     random_state: int = 42) -> dict:
    """
    Placebo / permutation test: randomly shuffle sentiment labels and re-run
    the univariate regression. If the true B is in the tail of the null
    distribution, the result is not a statistical artefact.

    Returns dict with:
      true_beta        -- observed OLS coefficient on sentiment_score_mean
      placebo_betas    -- list of beta from N permuted runs
      p_value_placebo  -- fraction of |placebo_beta| >= |true_beta|
    """
    if "sentiment_score_mean" not in df.columns or car_col not in df.columns:
        return {"error": "Missing sentiment_score_mean or CAR column"}

    valid = df[[car_col, "sentiment_score_mean"]].dropna()
    if len(valid) < 20:
        return {"error": f"Only {len(valid)} valid rows -- too few for permutation test"}

    y = valid[car_col].values
    x = valid["sentiment_score_mean"].values

    # True OLS B
    X_c = sm.add_constant(x)
    true_model = sm.OLS(y, X_c).fit()
    true_beta  = float(true_model.params[1])

    # Permutation distribution
    rng = np.random.default_rng(random_state)
    placebo_betas = []
    for _ in range(n_permutations):
        x_shuffled = rng.permutation(x)
        X_shuf     = sm.add_constant(x_shuffled)
        pla_model  = sm.OLS(y, X_shuf).fit()
        placebo_betas.append(float(pla_model.params[1]))

    placebo_arr = np.array(placebo_betas)
    p_val = float((np.abs(placebo_arr) >= np.abs(true_beta)).mean())

    log.info(f"Placebo test: true coef={true_beta:.5f}, "
             f"placebo p={p_val:.4f} ({n_permutations} permutations)")

    return {
        "true_beta":       round(true_beta, 6),
        "placebo_mean":    round(float(placebo_arr.mean()), 6),
        "placebo_std":     round(float(placebo_arr.std()),  6),
        "p_value_placebo": round(p_val, 4),
        "n_permutations":  n_permutations,
        "interpretation":  (
            "Result PASSES placebo test (true B in tail of null distribution)"
            if p_val < 0.10 else
            "Result FAILS placebo test (true B not distinguishable from noise)"
        ),
    }


def run_bootstrap_ci(df: pd.DataFrame, car_col: str,
                     n_bootstrap: int = 2000,
                     ci_level: float = 0.95,
                     random_state: int = 42) -> dict:
    """
    Bootstrap confidence intervals for the OLS coefficient on sentiment_score_mean.
    Resamples rows with replacement and refits OLS each time.

    Returns dict with lower_ci, upper_ci, bootstrap_std, and betas list.
    """
    if "sentiment_score_mean" not in df.columns or car_col not in df.columns:
        return {"error": "Missing columns"}

    valid = df[[car_col, "sentiment_score_mean"]].dropna().reset_index(drop=True)
    if len(valid) < 10:
        return {"error": "Insufficient observations for bootstrap"}

    rng = np.random.default_rng(random_state)
    betas = []
    n = len(valid)
    for _ in range(n_bootstrap):
        sample = valid.iloc[rng.integers(0, n, size=n)]
        y_s = sample[car_col].values
        X_s = sm.add_constant(sample["sentiment_score_mean"].values)
        try:
            m = sm.OLS(y_s, X_s).fit()
            betas.append(float(m.params[1]))
        except Exception:
            continue

    betas_arr = np.array(betas)
    alpha = 1 - ci_level
    lower = float(np.percentile(betas_arr, 100 * alpha / 2))
    upper = float(np.percentile(betas_arr, 100 * (1 - alpha / 2)))

    log.info(f"Bootstrap CI ({ci_level*100:.0f}%): [{lower:.5f}, {upper:.5f}] "
             f"over {len(betas)} valid resamples")

    return {
        "n_bootstrap":   n_bootstrap,
        "ci_level":      ci_level,
        "lower_ci":      round(lower, 6),
        "upper_ci":      round(upper, 6),
        "bootstrap_std": round(float(betas_arr.std()), 6),
        "bootstrap_mean":round(float(betas_arr.mean()), 6),
        "excludes_zero": lower > 0 or upper < 0,
    }


def print_robustness_summary(placebo: dict, bootstrap: dict):
    """Print a readable robustness summary."""
    print(f"\n{'-'*50}")
    print("Robustness Checks")
    print(f"{'-'*50}")

    if "error" not in placebo:
        print(f"\n  Placebo / Permutation Test ({placebo['n_permutations']} permutations)")
        print(f"    True B on sentiment      : {placebo['true_beta']:.5f}")
        print(f"    Placebo distribution     : mean={placebo['placebo_mean']:.5f}, "
              f"std={placebo['placebo_std']:.5f}")
        print(f"    p-value (permutation)    : {placebo['p_value_placebo']:.4f}")
        print(f"    -> {placebo['interpretation']}")
    else:
        print(f"  Placebo test skipped: {placebo['error']}")

    if "error" not in bootstrap:
        print(f"\n  Bootstrap CI ({bootstrap['ci_level']*100:.0f}%, {bootstrap['n_bootstrap']} resamples)")
        print(f"    B mean                   : {bootstrap['bootstrap_mean']:.5f}")
        print(f"    Bootstrap std            : {bootstrap['bootstrap_std']:.5f}")
        print(f"    {bootstrap['ci_level']*100:.0f}% CI                  : "
              f"[{bootstrap['lower_ci']:.5f}, {bootstrap['upper_ci']:.5f}]")
        excludes = bootstrap['excludes_zero']
        print(f"    CI excludes zero         : {'[OK] Yes' if excludes else '[ERR] No'}")
    else:
        print(f"  Bootstrap CI skipped: {bootstrap['error']}")


# -- Plots ----------------------------------------------------------------------

def plot_sentiment_vs_car(df: pd.DataFrame, car_col: str, out_dir: Path):
    """Scatter plot: sentiment score vs CAR with regression line."""
    if "sentiment_score_mean" not in df.columns:
        return

    # Use sector_original (sector was dummy-encoded by load_analysis_data)
    sec_col = "sector_original" if "sector_original" in df.columns else "sector"
    if sec_col not in df.columns:
        # No sector info available -- plot without sector coloring
        valid = df[[car_col, "sentiment_score_mean"]].dropna()
        if len(valid) < 10:
            return
        valid[sec_col] = "All"
    else:
        valid = df[[car_col, "sentiment_score_mean", sec_col]].dropna()
        if len(valid) < 10:
            return

    sector_colors = {
        "Banking":       "#2166ac",
        "Oil & Gas":     "#762a83",
        "Consumer Goods":"#1b7837",
        "Telecoms":      "#d6604d",
        "Industrial":    "#f4a582",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for sector, grp in valid.groupby(sec_col):
        color = sector_colors.get(sector, "grey")
        ax.scatter(grp["sentiment_score_mean"], grp[car_col],
                   label=sector, color=color, alpha=0.6, s=40, edgecolors="white", linewidth=0.4)

    # Regression line
    x = valid["sentiment_score_mean"].values
    y = valid[car_col].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, color="black", linewidth=1.5,
            linestyle="--", label=f"OLS fit (r={r:.2f}, p={p:.3f})")

    ax.axhline(0, color="grey", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("LLM Sentiment Score (mean)", fontsize=11)
    ax.set_ylabel(car_col, fontsize=11)
    ax.set_title(f"LLM Sentiment vs Post-Disclosure Abnormal Returns\n(NGX Listed Companies, 2020-2024)", fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    plt.tight_layout()
    fname = out_dir / f"sentiment_vs_{car_col.replace('[','').replace(']','').replace(',','_')}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot saved: {fname.name}")


def plot_car_by_group(group_tests_df: pd.DataFrame, car_col: str, out_dir: Path):
    """Bar chart of mean CAR by sentiment group with error bars."""
    groups_df = group_tests_df[
        group_tests_df["group"].isin(["Negative", "Neutral", "Positive"])
    ].copy()
    if groups_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"Negative": "#d6604d", "Neutral": "#92c5de", "Positive": "#4dac26"}
    x_pos  = range(len(groups_df))

    bars = ax.bar(
        x_pos,
        groups_df["mean_car"],
        color=[colors.get(g, "grey") for g in groups_df["group"]],
        alpha=0.85, width=0.5, edgecolor="white"
    )

    # Add significance stars
    for i, (_, row) in enumerate(groups_df.iterrows()):
        y_pos = row["mean_car"] + (0.002 if row["mean_car"] >= 0 else -0.004)
        ax.text(i, y_pos, row.get("sig", ""), ha="center", fontsize=13)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{g}\n(n={n})" for g, n in zip(groups_df["group"], groups_df["n"])],
        fontsize=10
    )
    ax.set_ylabel(f"Mean {car_col}", fontsize=11)
    ax.set_title(f"Mean Abnormal Returns by LLM Sentiment Group\n(NGX, 2020-2024; *** p<0.01, ** p<0.05, * p<0.10)", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fname = out_dir / f"car_by_group_{car_col.replace('[','').replace(']','').replace(',','_')}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot saved: {fname.name}")


def plot_event_window(ar_path: Path, out_dir: Path):
    """Average AR across event window days (-5 to +5)."""
    if not ar_path.exists():
        return
    ar_df = pd.read_csv(ar_path)
    if "day_relative" not in ar_df or "AR" not in ar_df:
        return

    window = ar_df[ar_df["day_relative"].between(-5, 5)]
    daily  = window.groupby("day_relative")["AR"].agg(["mean","sem","count"]).reset_index()
    daily["ci95"] = 1.96 * daily["sem"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(daily["day_relative"], daily["mean"], color="#4292c6", alpha=0.7,
           width=0.6, label="Mean AR")
    ax.errorbar(daily["day_relative"], daily["mean"], yerr=daily["ci95"],
                fmt="none", color="black", capsize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.6, label="Disclosure day")
    ax.set_xlabel("Trading Days Relative to Disclosure", fontsize=11)
    ax.set_ylabel("Mean Abnormal Return", fontsize=11)
    ax.set_title("Average Abnormal Returns Around Disclosure Date\n(NGX, All Companies 2020-2024)", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = out_dir / "event_window_avg_ar.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot saved: {fname.name}")


def plot_car_by_sector(df: pd.DataFrame, car_col: str, out_dir: Path):
    """Boxplot of CARs by sector."""
    if "sector_original" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    sectors = df["sector_original"].unique()
    data    = [df[df["sector_original"] == s][car_col].dropna().values for s in sectors]
    bp      = ax.boxplot(data, labels=sectors, patch_artist=True, notch=False,
                          medianprops=dict(color="black", linewidth=2))
    colors = ["#2166ac","#762a83","#1b7837","#d6604d","#f4a582"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sector", fontsize=11)
    ax.set_ylabel(car_col, fontsize=11)
    ax.set_title(f"Distribution of Abnormal Returns by Sector\n(NGX, 2020-2024)", fontsize=11)
    plt.tight_layout()
    fname = out_dir / "car_by_sector.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    log.info(f"  Plot saved: {fname.name}")


# -- LaTeX Tables ---------------------------------------------------------------

def latex_regression_table(results: list[dict], car_col: str) -> str:
    """Generate LaTeX regression table (Table 4 in paper)."""
    # Collect all unique variables across models
    key_vars = ["sentiment_score_mean", "guidance_score_mean", "guidance_rate",
                "weighted_score_mean", "net_sentiment", "guidance_adj_sentiment",
                "beta", "post_fx_unif"]

    model_labels = [r.get("label","").replace("M","Model ") for r in results if "error" not in r]
    valid_results = [r for r in results if "error" not in r]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Regression of Post-Disclosure Abnormal Returns on LLM Sentiment Scores (NGX 2020--2024)}",
        f"\\label{{tab:regression_{car_col.replace('[','').replace(']','').replace(',','_')}}}",
        r"\begin{tabular}{l" + "c" * len(valid_results) + "}",
        r"\hline",
        r"\textbf{Variable} & " + " & ".join([f"\\textbf{{{l}}}" for l in model_labels]) + r" \\",
        r"\hline",
    ]

    for var in key_vars:
        coef_row  = [var.replace("_", r"\_")]
        se_row    = [""]
        has_data  = False
        for r in valid_results:
            coefs = r.get("coefficients", {})
            if var in coefs:
                c = coefs[var]
                coef_row.append(f"{c['coef']:.4f}{c['sig']}")
                se_row.append(f"({c['std_err']:.4f})")
                has_data = True
            else:
                coef_row.append("--")
                se_row.append("")
        if has_data:
            lines.append(" & ".join(coef_row) + r" \\")
            lines.append(" & ".join(se_row) + r" \\")

    lines += [r"\hline"]

    # Stats rows
    for stat_label, stat_key in [
        ("Observations", "n_obs"),
        ("$R^2$", "r_squared"),
        ("Adj. $R^2$", "adj_r2"),
    ]:
        row = [stat_label]
        for r in valid_results:
            val = r.get(stat_key, "")
            row.append(str(val) if val != "" else "--")
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\hline",
        r"\multicolumn{" + str(len(valid_results)+1) + r"}{l}{\textit{Notes: HAC standard errors in parentheses. }} \\",
        r"\multicolumn{" + str(len(valid_results)+1) + r"}{l}{\textit{*** p$<$0.01, ** p$<$0.05, * p$<$0.10. Sentiment from LLM 5-shot combined prompts.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_group_test_table(group_df: pd.DataFrame, car_col: str) -> str:
    """Generate cross-sectional group test table (Table 5 in paper)."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Post-Disclosure Abnormal Returns by LLM Sentiment Group (NGX 2020--2024)}",
        r"\label{tab:group_tests}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Group} & \textbf{N} & \textbf{Mean " + car_col + r"} & \textbf{Median} & \textbf{t-stat} & \textbf{p-value} & \textbf{Sig} \\",
        r"\hline",
    ]
    for _, row in group_df.iterrows():
        mean_car = f"{row['mean_car']:.4f}" if pd.notna(row.get('mean_car')) else "--"
        median   = f"{row['median_car']:.4f}" if pd.notna(row.get('median_car')) else "--"
        t_stat   = f"{row['t_stat']:.3f}"     if pd.notna(row.get('t_stat'))    else "--"
        p_val    = f"{row['p_value']:.3f}"    if pd.notna(row.get('p_value'))   else "--"
        lines.append(
            f"{row['group']} & {row['n']} & {mean_car} & {median} & {t_stat} & {p_val} & {row.get('sig','')} \\\\"
        )
    lines += [
        r"\hline",
        r"\multicolumn{7}{l}{\textit{Notes: t-statistics test H$_0$: mean CAR = 0. Welch t-test for group differences.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# -- Main -----------------------------------------------------------------------

def run_analysis(car_col: str = "CAR[0,+3]", make_plots: bool = False):
    """Run full return-sentiment regression analysis."""
    log.info(f"\n{'='*60}")
    log.info(f"NGX-FND Return-Sentiment Analysis")
    log.info(f"Dependent variable: {car_col}")
    log.info(f"{'='*60}")

    df = load_analysis_data(car_col)
    if df is None:
        return

    # Store original sector for plots before dummying
    if "sector" in pd.read_csv(PRICES_DIR / "abnormal_returns.csv").columns:
        raw_df = pd.read_csv(PRICES_DIR / "abnormal_returns.csv", parse_dates=["event_date"])
        df["sector_original"] = raw_df["sector"].values[:len(df)]

    # -- Descriptive stats --
    print(f"\n{'-'*50}")
    print("Descriptive Statistics")
    print(f"{'-'*50}")
    print(f"N events:         {len(df)}")
    print(f"N companies:      {df['ticker'].nunique() if 'ticker' in df else '?'}")
    if car_col in df:
        print(f"Mean {car_col}:  {df[car_col].mean():.4f}")
        print(f"Median {car_col}:{df[car_col].median():.4f}")
        print(f"Std {car_col}:   {df[car_col].std():.4f}")
    if "sentiment_score_mean" in df:
        print(f"Mean sentiment:   {df['sentiment_score_mean'].mean():.3f}")
        print(f"% positive events:{(df['sentiment_score_mean'] > 0.2).mean():.2%}")
        print(f"% negative events:{(df['sentiment_score_mean'] < -0.2).mean():.2%}")

    # -- Regression models --
    print(f"\n{'-'*50}")
    print("Regression Results (HAC standard errors)")
    print(f"{'-'*50}")

    results = run_all_models(df, car_col)
    for r in results:
        if "error" in r:
            print(f"\n[{r.get('label','')}] ERROR: {r['error']}")
            continue
        print(f"\n{r['label']}")
        print(f"  N={r['n_obs']} | R2={r['r_squared']:.4f} | Adj-R2={r['adj_r2']:.4f}")
        for var, stats_d in r["coefficients"].items():
            if var == "const":
                continue
            print(f"  {var:<35} coef={stats_d['coef']:>9.5f}  "
                  f"t={stats_d['t_stat']:>6.3f}  "
                  f"p={stats_d['p_value']:.3f} {stats_d['sig']}")

    # -- Group tests --
    print(f"\n{'-'*50}")
    print("Cross-Sectional Group Tests")
    print(f"{'-'*50}")
    group_df = run_group_tests(df, car_col)
    if not group_df.empty:
        print(group_df.to_string(index=False))

    # -- Robustness checks --
    raw_df_for_robust = load_analysis_data.__wrapped__(car_col) \
        if hasattr(load_analysis_data, "__wrapped__") else \
        pd.read_csv(PRICES_DIR / "abnormal_returns.csv", parse_dates=["event_date"])
    # Use the same df that went into regressions (before dummy encoding breaks numeric cols)
    placebo   = run_placebo_test(df, car_col)
    bootstrap = run_bootstrap_ci(df, car_col)
    print_robustness_summary(placebo, bootstrap)

    # Save robustness results
    robustness = {"placebo_test": placebo, "bootstrap_ci": bootstrap}
    import json as _json
    with open(ANALYSIS_DIR / "robustness_checks.json", "w") as f:
        _json.dump(robustness, f, indent=2, default=str)
    log.info(f"  Robustness checks saved: {ANALYSIS_DIR / 'robustness_checks.json'}")

    # -- Save results --
    if results:
        # Flatten for CSV
        rows = []
        for r in results:
            if "error" in r:
                continue
            base = {k: v for k, v in r.items() if k != "coefficients"}
            for var, s in r.get("coefficients", {}).items():
                rows.append({**base, "variable": var, **s})
        pd.DataFrame(rows).to_csv(ANALYSIS_DIR / "regression_results.csv", index=False)

    if not group_df.empty:
        group_df.to_csv(ANALYSIS_DIR / "group_tests.csv", index=False)

    # -- LaTeX tables --
    latex_out = []
    if results:
        latex_out.append("% === TABLE 4: Regression Results ===")
        latex_out.append(latex_regression_table(results, car_col))
    if not group_df.empty:
        latex_out.append("\n% === TABLE 5: Group Tests ===")
        latex_out.append(latex_group_test_table(group_df, car_col))

    if latex_out:
        tex_path = ANALYSIS_DIR / "paper_tables.tex"
        with open(tex_path, "w") as f:
            f.write("\n\n".join(latex_out))
        log.info(f"\n[doc] LaTeX tables saved: {tex_path}")

    # -- Plots --
    if make_plots:
        log.info("\nGenerating figures...")
        run_dir = FIGURES_DIR / f"returns_{datetime.now().strftime('%Y%m%d')}"
        run_dir.mkdir(exist_ok=True)
        plot_sentiment_vs_car(df, car_col, run_dir)
        plot_car_by_group(group_df, car_col, run_dir)
        plot_event_window(PRICES_DIR / "event_windows.csv", run_dir)
        if "sector_original" in df:
            plot_car_by_sector(df, car_col, run_dir)
        log.info(f"[chart] All figures saved to: {run_dir}")

    log.info(f"\n[OK] Analysis complete. Results in: {ANALYSIS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="NGX-FND Return-Sentiment Analysis")
    parser.add_argument("--car",   default="CAR[0,+3]",
                        choices=list(["CAR[-1,+1]","CAR[0,+1]","CAR[0,+3]",
                                      "CAR[0,+5]","CAR[-1,+3]","CAR[-1,+5]"]),
                        help="CAR window to use as dependent variable")
    parser.add_argument("--plots", action="store_true", help="Generate all figures")
    args = parser.parse_args()
    run_analysis(car_col=args.car, make_plots=args.plots)


if __name__ == "__main__":
    main()

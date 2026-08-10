"""tools.py — the tools exposed to the LLM: run_sql, run_analysis, make_chart.

All dataset access goes through a read-only, validated SQL path. Nothing here is
domain-specific; the tools operate on whatever schema was loaded.
"""

from __future__ import annotations

import re
import io
import base64
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# run_sql — safe, read-only SQL execution
# ---------------------------------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|"
    r"attach|detach|pragma|vacuum|reindex|grant|revoke)\b",
    re.IGNORECASE,
)


class SQLValidationError(Exception):
    pass


def _validate_sql(query: str) -> str:
    q = query.strip().rstrip(";").strip()
    if not q:
        raise SQLValidationError("Empty query.")
    # single statement only
    if ";" in q:
        raise SQLValidationError("Only a single statement is allowed (no ';').")
    low = q.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise SQLValidationError("Only SELECT / WITH queries are allowed.")
    if _FORBIDDEN.search(q):
        raise SQLValidationError("Query contains a forbidden (write/DDL) keyword.")
    return q


def _enforce_limit(query: str, max_rows: int) -> str:
    if re.search(r"\blimit\b", query, re.IGNORECASE):
        return query
    return f"{query}\nLIMIT {max_rows}"


def run_sql(
    query: str,
    db_path: str,
    max_rows: int = 5000,
    timeout_s: int = 15,
) -> dict[str, Any]:
    """Execute a read-only SELECT against the SQLite DB.

    Returns {"columns": [...], "rows": [...], "row_count": n, "sql": "..."}.
    Raises SQLValidationError on unsafe input.
    """
    q = _validate_sql(query)
    q = _enforce_limit(q, max_rows)

    engine = create_engine(
        f"sqlite:///file:{db_path}?mode=ro&uri=true",
        connect_args={"uri": True, "timeout": timeout_s},
    )
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(f"PRAGMA query_only = ON")
            df = pd.read_sql(text(q), conn)
    finally:
        engine.dispose()

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "row_count": int(len(df)),
        "sql": q,
        "_df": df,  # kept for internal chaining; stripped before sending to LLM
    }


# ---------------------------------------------------------------------------
# run_analysis — statistics on a query result
# ---------------------------------------------------------------------------

def _as_df(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict) and "_df" in data:
        return data["_df"]
    if isinstance(data, dict) and "rows" in data:
        return pd.DataFrame(data["rows"])
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise ValueError("Unsupported data payload for analysis.")


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def run_analysis(data: Any, method: str, **kwargs) -> dict[str, Any]:
    """Run a statistical method on query results.

    method ∈ {trend, correlation, outliers, group_compare, describe}.
    Returns a small JSON-serializable summary plus a plain-language note.
    """
    df = _as_df(data)
    method = (method or "").lower()

    if df.empty:
        return {"method": method, "note": "No data to analyze."}

    if method == "describe":
        desc = df.describe(include="all").fillna("").to_dict()
        return {"method": method, "summary": desc,
                "note": f"Descriptive stats over {len(df)} rows."}

    if method == "correlation":
        from scipy import stats
        nums = _numeric_cols(df)
        if len(nums) < 2:
            return {"method": method, "note": "Need at least two numeric columns."}
        x, y = kwargs.get("x", nums[0]), kwargs.get("y", nums[1])
        sub = df[[x, y]].dropna()
        r, p = stats.pearsonr(sub[x], sub[y])
        strength = ("strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak")
        direction = "positive" if r > 0 else "negative"
        return {
            "method": method, "x": x, "y": y,
            "pearson_r": round(float(r), 4), "p_value": round(float(p), 6),
            "note": f"{strength} {direction} correlation between {x} and {y} "
                    f"(r={r:.2f}, p={p:.3g}).",
        }

    if method == "trend":
        import statsmodels.api as sm
        nums = _numeric_cols(df)
        if not nums:
            return {"method": method, "note": "No numeric column to trend."}
        y_col = kwargs.get("y", nums[-1])
        y = pd.to_numeric(df[y_col], errors="coerce").dropna().reset_index(drop=True)
        x = np.arange(len(y))
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        slope = float(model.params[1])
        p = float(model.pvalues[1])
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        sig = "statistically significant" if p < 0.05 else "not statistically significant"
        return {
            "method": method, "y": y_col, "slope": round(slope, 6),
            "p_value": round(p, 6), "r_squared": round(float(model.rsquared), 4),
            "note": f"{y_col} shows a {direction} trend "
                    f"(slope={slope:.3g}, {sig}, R²={model.rsquared:.2f}).",
        }

    if method == "outliers":
        nums = _numeric_cols(df)
        if not nums:
            return {"method": method, "note": "No numeric column for outlier detection."}
        col = kwargs.get("column", nums[-1])
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lo) | (s > hi)]
        return {
            "method": method, "column": col,
            "n_outliers": int(len(outliers)),
            "bounds": [round(float(lo), 4), round(float(hi), 4)],
            "outlier_values": [round(float(v), 4) for v in outliers.head(20)],
            "note": f"Found {len(outliers)} IQR outliers in {col} "
                    f"(outside [{lo:.2f}, {hi:.2f}]).",
        }

    if method == "group_compare":
        from scipy import stats
        group_col = kwargs.get("group")
        value_col = kwargs.get("value")
        nums = _numeric_cols(df)
        if not group_col or not value_col:
            cats = [c for c in df.columns if c not in nums]
            group_col = group_col or (cats[0] if cats else None)
            value_col = value_col or (nums[0] if nums else None)
        if not group_col or not value_col:
            return {"method": method, "note": "Could not infer group/value columns."}
        groups = [g[value_col].dropna().values for _, g in df.groupby(group_col)]
        groups = [g for g in groups if len(g) > 1]
        result: dict[str, Any] = {
            "method": method, "group": group_col, "value": value_col,
            "group_means": df.groupby(group_col)[value_col].mean().round(4).to_dict(),
        }
        if len(groups) == 2:
            t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            result.update(test="welch_t", statistic=round(float(t), 4),
                          p_value=round(float(p), 6))
        elif len(groups) > 2:
            f, p = stats.f_oneway(*groups)
            result.update(test="anova", statistic=round(float(f), 4),
                          p_value=round(float(p), 6))
        sig = result.get("p_value")
        result["note"] = (
            f"Compared {value_col} across {group_col}"
            + (f"; {result.get('test')} p={sig:.3g} "
               f"({'significant' if sig is not None and sig < 0.05 else 'not significant'})."
               if sig is not None else ".")
        )
        return result

    return {"method": method, "note": f"Unknown method '{method}'."}


# ---------------------------------------------------------------------------
# make_chart — matplotlib PNG (base64) + a lightweight spec
# ---------------------------------------------------------------------------

def make_chart(
    data: Any,
    chart_type: str = "bar",
    x: str | None = None,
    y: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Render a chart from query results. Returns base64 PNG + echo of the spec."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _as_df(data)
    if df.empty:
        return {"chart_type": chart_type, "note": "No data to chart.", "image_base64": None}

    nums = _numeric_cols(df)
    cats = [c for c in df.columns if c not in nums]

    # sensible auto-pick of axes
    if x is None:
        x = cats[0] if cats else df.columns[0]
    if y is None:
        y = nums[0] if nums else (df.columns[1] if len(df.columns) > 1 else df.columns[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        plot_df = df.head(50)
        if chart_type == "line":
            ax.plot(plot_df[x], plot_df[y], marker="o")
        elif chart_type == "scatter":
            ax.scatter(plot_df[x], plot_df[y])
        else:  # bar (default)
            ax.bar(plot_df[x].astype(str), plot_df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f"{y} by {x}")
        if plot_df[x].astype(str).map(len).max() > 6 or chart_type == "bar":
            plt.xticks(rotation=45, ha="right")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        plt.close(fig)
        return {"chart_type": chart_type, "note": f"Chart error: {e}", "image_base64": None}

    return {
        "chart_type": chart_type, "x": x, "y": y,
        "title": title or f"{y} by {x}",
        "image_base64": b64,
        "note": f"Rendered {chart_type} chart of {y} vs {x}.",
    }

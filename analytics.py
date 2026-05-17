import re
import pandas as pd
from config import ANALYTICAL_KEYWORDS


def is_analytical(query: str) -> bool:
    tokens = set(query.lower().split())
    return bool(tokens & ANALYTICAL_KEYWORDS)


def _top_n(df: pd.DataFrame, fname: str, n: int, q: str) -> str:
    rev = df.groupby("Product")["Revenue"].sum() if "Revenue" in df.columns else None
    qty = df.groupby("Product")["Quantity Ordered"].sum() if "Quantity Ordered" in df.columns else None

    if "quantity" in q and qty is not None:
        tbl = qty.nlargest(n).reset_index().rename(columns={"Quantity Ordered": "Total Qty"})
    elif rev is not None:
        tbl = rev.nlargest(n).reset_index().rename(columns={"Revenue": "Total Revenue ($)"})
        tbl["Total Revenue ($)"] = tbl["Total Revenue ($)"].round(2)
    else:
        return ""
    return f"[{fname}]\n{tbl.to_string(index=False)}"


def _total_revenue(df: pd.DataFrame, fname: str) -> str:
    if "Revenue" not in df.columns:
        return ""
    return f"[{fname}] Total revenue: ${df['Revenue'].sum():,.2f}"


def _average(df: pd.DataFrame, fname: str) -> str:
    parts = []
    if "Revenue" in df.columns:
        parts.append(f"[{fname}] Average order revenue: ${df['Revenue'].mean():,.2f}")
    if "Quantity Ordered" in df.columns:
        parts.append(f"[{fname}] Average quantity ordered: {df['Quantity Ordered'].mean():.1f}")
    return "\n".join(parts)


def _best_worst(df: pd.DataFrame, fname: str, best: bool) -> str:
    if "Revenue" not in df.columns:
        return ""
    rev_by_product = df.groupby("Product")["Revenue"].sum()   # computed once
    label = "Best selling" if best else "Lowest revenue"
    product = rev_by_product.idxmax() if best else rev_by_product.idxmin()
    val = rev_by_product.max() if best else rev_by_product.min()
    return f"[{fname}] {label} product: {product} (${val:,.2f})"


def _order_count(df: pd.DataFrame, fname: str) -> str:
    return f"[{fname}] Total orders: {len(df):,}"


def _product_list(df: pd.DataFrame, fname: str) -> str:
    products = df["Product"].unique().tolist()
    preview = ", ".join(products[:20]) + ("…" if len(products) > 20 else "")
    return f"[{fname}] Products in dataset ({len(products)}):\n{preview}"


def run_analytical_query(query: str, dfs: dict[str, pd.DataFrame]) -> str | None:
    q = query.lower()
    results = []

    for fname, df in dfs.items():
        if df.empty:
            continue

        top_match = re.search(r"top\s+(\d+)", q)
        if top_match:
            result = _top_n(df, fname, int(top_match.group(1)), q)
        elif any(kw in q for kw in ("total revenue", "total sales")):
            result = _total_revenue(df, fname)
        elif any(kw in q for kw in ("average", "avg", "mean")):
            result = _average(df, fname)
        elif any(kw in q for kw in ("best", "highest", "most")):
            result = _best_worst(df, fname, best=True)
        elif any(kw in q for kw in ("worst", "lowest", "least")):
            result = _best_worst(df, fname, best=False)
        elif "count" in q or "how many orders" in q:
            result = _order_count(df, fname)
        elif "product" in q:
            result = _product_list(df, fname)
        else:
            result = ""

        if result:
            results.append(result)

    return "\n\n".join(results) if results else None
"""Read-only housing audit: approved decisions and review candidates stay separate."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {
    "Bedrooms >= 2 and area < 400": 7,
    "Bedrooms >= 8 and area < 1200": 3,
    "Vacant Land": 16,
    "Single Family with zero bedrooms and bathrooms": 115,
    "Single Family with bedrooms but zero bathrooms": 117,
    "Bedrooms >= 2 and 400 <= area < 500": 42,
}


def load_data():
    """Load the original CSV; never write changes back to it."""
    csv_path = PROJECT_ROOT / "data" / "raw" / "canada_housing.csv"
    df = pd.read_csv(csv_path, encoding="latin1")
    return df


def audit_overview(df):
    """Return column types and missingness; duplicate copies are counted separately."""
    return pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean().mul(100).round(2),
    })


def match_property(df, city, province, property_type, price, area, bedrooms,
                   bathrooms, acreage=None):
    """Match every supplied attribute, including all duplicate copies."""
    match = (
        df["City"].eq(city) & df["Province"].eq(province)
        & df["Property Type"].eq(property_type) & df["Price"].eq(price)
        & df["Square Footage"].eq(area) & df["Bedrooms"].eq(bedrooms)
        & df["Bathrooms"].eq(bathrooms)
    )
    if acreage is not None:
        match = match & df["Acreage"].eq(acreage)
    return match.fillna(False)


def flag_approved_removals(df):
    """Identify approved removal decisions without removing any rows."""
    bedrooms, bathrooms, area = df["Bedrooms"], df["Bathrooms"], df["Square Footage"]
    return pd.DataFrame({
        "Exact duplicate copy (keep first)": df.duplicated(keep="first"),
        "Bedrooms >= 2 and area < 400": bedrooms.ge(2) & area.lt(400),
        "Bedrooms >= 8 and area < 1200": bedrooms.ge(8) & area.lt(1200),
        "Vacant Land": df["Property Type"].eq("Vacant Land"),
        "Single Family with zero bedrooms and bathrooms": (
            df["Property Type"].eq("Single Family") & bedrooms.eq(0) & bathrooms.eq(0)
        ),
        "Single Family with bedrooms but zero bathrooms": (
            df["Property Type"].eq("Single Family") & bedrooms.gt(0) & bathrooms.eq(0)
        ),
        "Bedrooms >= 2 and 400 <= area < 500": bedrooms.ge(2) & area.ge(400) & area.lt(500),
        "Investigated Allison property": match_property(
            df, "Allison", "NB", "Single Family", 699900, 147239, 3, 3),
        "Investigated Kitchener property": match_property(
            df, "Kitchener", "ON", "MultiFamily", 1625000, 3318, 0, 0, 4773),
        "Investigated Temiskaming Shores property": match_property(
            df, "Temiskaming Shores (Haileybury)", "ON", "MultiFamily",
            4495000, 24500, 37, 0),
        "Investigated Langley property": match_property(
            df, "Langley", "BC", "Single Family", 8990000, 864, 2, 0, 5.5),
    }, index=df.index).fillna(False).astype(bool)


def audit_numeric_validity(df):
    """Flag invalid numeric values and specification anomalies for review only."""
    columns = ["Price", "Bedrooms", "Bathrooms", "Square Footage", "Acreage"]
    checks = {f"Negative {column}": df[column].lt(0) for column in columns}
    bedrooms, bathrooms, area = df["Bedrooms"], df["Bathrooms"], df["Square Footage"]
    checks.update({
        "Zero price": df["Price"].eq(0),
        "Zero floor area": area.eq(0),
        "Bathrooms > twice bedrooms": bathrooms.gt(bedrooms * 2),
        "Area > 10000": area.gt(10000),
        "Bedrooms >= 18": bedrooms.ge(18),
        "Bathrooms >= 17": bathrooms.ge(17),
        "Price >= 5000000": df["Price"].ge(5000000),
        "Acreage > 100": df["Acreage"].gt(100),
        "Acreage > 500 (subset)": df["Acreage"].gt(500),
    })
    return pd.DataFrame(checks, index=df.index).fillna(False).astype(bool)


def flag_review_candidates(df, approved):
    """Exclude approved-removal rows from all review-only flags."""
    _, categorical_flags = audit_categorical_features(df)
    checks = pd.concat([
        audit_numeric_validity(df), categorical_flags, audit_location(df)
    ], axis=1)
    checks.loc[approved.any(axis=1), :] = False
    return checks


def categorical_columns(df):
    """Find text, categorical, and Boolean features from the loaded schema."""
    return df.select_dtypes(include=["object", "string", "category", "bool"]).columns


def audit_categorical_features(df):
    """Profile every original category and flag formatting without normalizing data."""
    profiles, checks = [], {}
    for column in categorical_columns(df):
        values = df[column]
        text = values.astype("string")
        stripped = text.str.strip()
        blank = text.eq("").fillna(False)
        whitespace = stripped.eq("").fillna(False) & ~blank
        spaces = text.ne(stripped).fillna(False)
        # Temporary case-folded keys identify variants; original values are retained.
        variants = pd.DataFrame({"key": stripped.str.casefold(), "spelling": stripped})
        variant_counts = variants.groupby("key")["spelling"].nunique()
        case_variation = variants["key"].map(variant_counts).gt(1).fillna(False)
        checks[f"{column}: blank or whitespace-only"] = blank | whitespace
        checks[f"{column}: surrounding spaces"] = spaces
        checks[f"{column}: case variants"] = case_variation
        counts = values.value_counts(dropna=False, sort=False)
        profile = counts.rename_axis("category").reset_index(name="frequency")
        category_text = profile["category"].astype("string")
        category_stripped = category_text.str.strip()
        profile.insert(0, "column", column)
        profile["percentage"] = profile["frequency"].div(len(df)).mul(100).round(4)
        profile["is_missing"] = profile["category"].isna()
        profile["is_blank_string"] = category_text.eq("").fillna(False)
        profile["is_whitespace_only"] = (
            category_stripped.eq("").fillna(False) & ~profile["is_blank_string"]
        )
        profile["has_surrounding_spaces"] = category_text.ne(category_stripped).fillna(False)
        profile["has_case_variants"] = (
            category_stripped.str.casefold().map(variant_counts).gt(1).fillna(False)
        )
        profile["non_missing_count"] = values.notna().sum()
        profile["missing_count"] = values.isna().sum()
        profile["missing_percentage"] = round(values.isna().mean() * 100, 4)
        profile["distinct_non_missing"] = values.nunique()
        profiles.append(profile)
    return pd.concat(profiles, ignore_index=True), pd.DataFrame(checks, index=df.index)


def audit_location(df):
    """Check missing locations and global coordinate validity; infer no country bounds."""
    checks = {}
    for column in ["Province", "City"]:
        values = df[column].astype("string")
        checks[f"{column}: missing or blank"] = (
            values.isna() | values.str.strip().eq("").fillna(False)
        )
    coordinates = {}
    for column, limit in [("Latitude", 90), ("Longitude", 180)]:
        values = df[column]
        numeric = pd.to_numeric(values, errors="coerce")
        coordinates[column] = numeric
        checks[f"{column}: missing"] = values.isna()
        checks[f"{column}: non-numeric"] = values.notna() & numeric.isna()
        checks[f"{column}: outside global range"] = numeric.lt(-limit) | numeric.gt(limit)
    checks["Coordinates: (0, 0)"] = (
        coordinates["Latitude"].eq(0) & coordinates["Longitude"].eq(0)
    )
    return pd.DataFrame(checks, index=df.index).fillna(False).astype(bool)


def analyze_price_relationships(df):
    """Describe associations on supplied analysis rows; never determine flags."""
    price = pd.to_numeric(df["Price"], errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    results = []
    for column in df.select_dtypes(include="number").columns:
        if column == "Price":
            continue
        feature = pd.to_numeric(df[column], errors="coerce").replace(
            [float("inf"), -float("inf")], float("nan")
        )
        pairs = pd.DataFrame({"feature": feature, "price": price}).dropna()
        meaningful = len(pairs) >= 2 and pairs["feature"].nunique() > 1 and pairs["price"].nunique() > 1
        results.append({
            "feature": column, "analysis_type": "numeric_correlation",
            "observation_count": len(pairs),
            "pearson": pairs["feature"].corr(pairs["price"]) if meaningful else None,
            # Pearson on ranks is Spearman, without an optional SciPy dependency.
            "spearman": pairs["feature"].rank().corr(pairs["price"].rank()) if meaningful else None,
        })
    for column in categorical_columns(df):
        grouped = pd.DataFrame({"category": df[column], "price": price}).groupby(
            "category", dropna=False, observed=True, sort=False
        )["price"].agg(property_count="size", observation_count="count",
                       mean_price="mean", median_price="median").reset_index()
        grouped["feature"] = column
        grouped["analysis_type"] = "categorical_price_summary"
        grouped["is_missing_category"] = grouped["category"].isna()
        results.extend(grouped.to_dict("records"))
    return pd.DataFrame(results).reindex(columns=[
        "feature", "analysis_type", "category", "is_missing_category", "property_count",
        "observation_count", "mean_price", "median_price", "pearson", "spearman",
    ])


def flagged_records(df, flags):
    """Copy flagged records with their original indices and all matching reasons."""
    selected = flags.any(axis=1)
    records = df.loc[selected].copy()
    records["reasons"] = flags.loc[selected].apply(
        lambda row: "; ".join(row.index[row]), axis=1
    )
    return records


def generate_report(df, overview, approved, review, report_dir):
    """Save a summary and flagged records without cleaning data."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    categorical_profile, categorical_flags = audit_categorical_features(df)
    location_flags = audit_location(df)
    analysis_df = df.loc[~approved.any(axis=1)].copy()
    relationships = analyze_price_relationships(analysis_df)
    counts = pd.concat({"approved_removal": approved.sum(), "review_only": review.sum()})
    records = []
    for status, flags in [("approved_removal", approved), ("review_only", review)]:
        flagged = flagged_records(df, flags)
        flagged["audit_status"] = status
        records.append(flagged)
    pd.concat(records).to_csv(report_dir / "flagged_records.csv", index_label="source_index")
    lines = [
        "Canadian housing audit (read-only)",
        f"Dataset: {len(df):,} rows, {len(df.columns)} columns",
        f"Exact duplicate copies, keeping first: {df.duplicated().sum():,}",
        f"Unique approved-removal rows: {approved.any(axis=1).sum():,}",
        f"Unique review-only rows: {review.any(axis=1).sum():,}",
        "Counts overlap within categories; unique totals count each source row once.",
        "No rows or values were changed. Missing values are reported, not filled.",
        "\nMissing values by column:",
        overview[["missing_count", "missing_percent"]].to_string(),
        "\nNumeric validation findings (all raw rows):",
        audit_numeric_validity(df).sum().to_string(),
        f"Categorical columns inspected: {len(categorical_columns(df))}",
        f"Rows with categorical blank/formatting issues (all raw rows): {categorical_flags.any(axis=1).sum():,}",
        "Categorical findings (all raw rows; zero-count rules omitted):",
        categorical_flags.sum().loc[lambda counts: counts.gt(0)].to_string(),
        "Location validation findings (all raw rows):", location_flags.sum().to_string(),
        "Price relationships exclude approved removals: Pearson/Spearman correlations and category price summaries.",
        f"Rows used for price relationship analysis: {len(analysis_df):,}",
        "Associations are descriptive, not causal or removal rules; no model was trained.",
        "Missing amenities remain unknown; rare categories are not automatically invalid.",
        "\nPer-rule counts:", counts.to_string(), "\nExpected finding checks:",
    ]
    for rule, expected in EXPECTED_COUNTS.items():
        actual = int(approved[rule].sum())
        status = "MATCH" if actual == expected else "MISMATCH"
        lines.append(f"{status}: {rule}: expected {expected}, observed {actual}")
    for rule in ["Investigated Allison property", "Investigated Kitchener property",
                 "Investigated Temiskaming Shores property", "Investigated Langley property"]:
        count = int(approved[rule].sum())
        lines.append(f"Specific selector: {rule}: {count} rows (including duplicate copies)")
        if count == 0:
            lines.append(f"MISMATCH: no records found for {rule}")
    london = match_property(df, "London", "ON", "MultiFamily", 3450888, 16000, 12, 26)
    lines.append(f"London investigated record: {london.sum()} matched; "
                 f"{(london & approved.any(axis=1)).sum()} independently approved for removal")
    summary = "\n".join(lines) + "\n"
    (report_dir / "audit_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Reports saved to: {report_dir}")


def run_audit(report_dir=None):
    """Load, audit, and report; return Boolean flags with source indices intact."""
    df = load_data()
    overview = audit_overview(df)
    approved = flag_approved_removals(df)
    review = flag_review_candidates(df, approved)
    if report_dir is None:
        report_dir = PROJECT_ROOT / "reports" / "audit"
    generate_report(df, overview, approved, review, report_dir)
    return {"approved_removals": approved, "review_candidates": review}


if __name__ == "__main__":
    run_audit()

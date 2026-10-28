# Canada housing audit decisions

## Scope and final counts

Manual investigation was followed by automated, read-only validation in
[`src/audit_pipeline.py`](../../src/audit_pipeline.py). The audit records approved
exclusions separately from candidates requiring review; it does not clean data.

| Measure | Rows |
|---|---:|
| Original dataset (23 columns) | 44,896 |
| Exact duplicate copies, keeping the first occurrence | 4,082 |
| Unique approved-removal rows | 4,360 |
| Expected remaining rows after approved removals | 40,536 |
| Unique review-only records | 1,714 |

The 40,536 remaining rows include the review-only records. Price relationships
already use this analysis-only subset; other audits inspect the full raw dataset.
All six expected-count checks match the current findings.

## Approved removals

These are project-specific decisions from the investigation, not universal rules
about valid housing. Counts below refer to matches in the original dataset.

| Rule | Matches |
|---|---:|
| Exact duplicate copies, keeping the first | 4,082 |
| Bedrooms ≥ 2 and square footage < 400 | 7 |
| Bedrooms ≥ 8 and square footage < 1,200 | 3 |
| Property Type = Vacant Land | 16 |
| Single Family with zero bedrooms and zero bathrooms | 115 |
| Single Family with bedrooms > 0 and zero bathrooms | 117 |
| Bedrooms ≥ 2 and 400 ≤ square footage < 500 | 42 |
| Specifically investigated Allison property | 1 |
| Specifically investigated Kitchener property | 1 |
| Specifically investigated Temiskaming Shores property | 2 |
| Specifically investigated Langley property | 1 |

Zero-bathroom Single Family records are excluded rather than assuming a correct
bathroom count. The small multi-bedroom listings are excluded as a conservative
data-quality decision. Vacant Land is excluded following the manual investigation
of residential-attribute inconsistencies.

The individual selectors require all the listed attributes to match:

| City / province | Property type | Price | Sqft | Beds | Baths | Acreage requirement |
|---|---|---:|---:|---:|---:|---|
| Allison, NB | Single Family | 699,900 | 147,239 | 3 | 3 | Not required |
| Kitchener, ON | MultiFamily | 1,625,000 | 3,318 | 0 | 0 | 4,773 |
| Temiskaming Shores (Haileybury), ON | MultiFamily | 4,495,000 | 24,500 | 37 | 0 | Not required |
| Langley, BC | Single Family | 8,990,000 | 864 | 2 | 0 | 5.5 |

The Temiskaming Shores selector includes both copies. These selectors do not
exclude every listing in a city.

Rule counts overlap: a duplicate may also match a specification rule. The approved
total counts each original `source_index` once. The two most recently approved
groups added 141 unique rows: 109 + 34 − 2 overlapping rows. Approved and review-only
statuses are mutually exclusive.

## Retention and remaining review

Plausible expensive, large, rural, and unusual properties remain unless they match
an approved rule. High prices, large floor areas or acreages, extreme room counts,
and bathroom-to-bedroom ratios are review signals, not automatic exclusions.
Rare categories are not invalid merely because they are rare.

The investigated London, ON MultiFamily listing (Price 3,450,888; 16,000 sqft;
12 bedrooms; 26 bathrooms) remains: it matches no approved-removal rule.
Review-only records have not all been certified as correct; their flags remain
available for follow-up. Price associations are descriptive and never removal rules.

City capitalization variants affect 534 raw rows, including 521 remaining review-only
rows. Standardize equivalent city spellings during cleaning while preserving place
identity; no capitalization changes have been applied yet. The audit found no
blank or surrounding-whitespace issues, missing city/province values, invalid global
coordinate ranges, missing/non-numeric coordinates, or (0, 0) pairs. These checks do
not establish that every coordinate correctly identifies a Canadian property.

## Missing values

Missingness in the original dataset is concentrated in five columns:

| Column | Missing | Percentage |
|---|---:|---:|
| Basement | 29,934 | 66.67% |
| Exterior | 27,457 | 61.16% |
| Heating | 6,654 | 14.82% |
| Flooring | 29,680 | 66.11% |
| Roof | 34,844 | 77.61% |

The other 18 columns have no missing values in the loaded data. Treatment is
deferred to cleaning/preprocessing. Missing amenities remain unknown; they have not
been replaced with “No” or “None,” and no imputation strategy is approved here.

## Audit reports and next stage

- [`audit_summary.txt`](audit_summary.txt): dataset totals, missingness, numeric,
  categorical and location findings, rule counts, and expected-count checks.
- [`audit_findings.xlsx`](audit_findings.xlsx): Column Overview, Rule Counts,
  Categorical Profile, Price Relationships, and sectioned Flagged Records. Records
  can appear in multiple workbook sections when they match multiple categories.
- [`flagged_records.csv`](flagged_records.csv): one row per flagged source record,
  retaining all 23 original columns, `source_index`, `reasons`, and `audit_status`.

The raw dataset has not been modified and is not yet cleaned. The next stage is to
implement `cleaning.py` using these approved exclusions and city-capitalization
decisions, with missing-value treatment addressed explicitly. Preserve the raw file
and source traceability. Do not use Price-derived features, such as Price per Square
Footage, as model inputs. No cleaning or model training is implemented by this document.

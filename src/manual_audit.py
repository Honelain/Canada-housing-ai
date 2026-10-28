import pandas as pd

# Loading the original data to work through the checks
df = pd.read_csv("data/raw/canada_housing.csv", encoding="latin1")

# Starting with the dataset size, columns, and missing values
print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum())

missing_percent = df.isnull().mean() * 100

print(missing_percent.sort_values(ascending=False))
# Checking for repeated rows and looking at a few examples
print(df.duplicated().sum())
print(df[df.duplicated()].head()[["Province", "City", "Price", "Longitude", "Latitude", "Bedrooms", "Bathrooms"]])
print(df.describe().round(2)[["Square Footage", "Price", "Longitude", "Latitude", "Bedrooms", "Bathrooms""", "Acreage"]])
# Looking at the largest floor areas alongside the other property details
print(df["Square Footage"].nlargest(10).to_string(index=False))
print(df.nlargest(2, "Square Footage")[[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].to_string(index=False))

print(df.nlargest(10, "Square Footage")[[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].to_string(index=False))

# Taking a closer look at the Allison listings
print(df[df["City"] == "Allison"][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].to_string(index=False))

# Comparing floor areas by property type before judging the large values
print(df.groupby("Property Type")["Square Footage"].agg(
    ["count", "min", "median", "max"]
).sort_values("max", ascending=False))

# Checking which property types account for areas above 10,000 square feet
print((df["Square Footage"] > 10000).sum())
print(df[df["Square Footage"] > 10000]["Property Type"].value_counts())
print(df[
    (df["Square Footage"] > 10000) &
    (df["Property Type"] == "Single Family")
][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].sort_values("Square Footage", ascending=False).to_string(index=False))
print(df[
    (df["Square Footage"] > 10000) &
    (df["Property Type"] == "Single Family")
].describe().round(2))
# Checking the 147,239 sqft record against its room counts and amenities
print(df[df["Square Footage"] == 147239][["City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage",
    "Pool",
    "Balcony",
    "Garden"
]])
# Comparing price per square foot for the Allison listings as another check
print(df[df["City"] == "Allison"][[
    "Price",
    "Square Footage",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].assign(
    Price_per_sqft=lambda x: x["Price"] / x["Square Footage"]
).to_string(index=False))
print(df[df["City"] == "Allison"])
# Using the upper quantiles to put these floor areas in context
print(df["Square Footage"].quantile([0.90, 0.95, 0.99, 0.999]))
print(df[df["Square Footage"] == 147239][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage",
    "Latitude",
    "Longitude"
]].to_string(index=False))
print(df[df["City"] == "Allison"][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Acreage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Latitude",
    "Longitude"
]].to_string(index=False))
# Looking at unusually high bedroom counts and the properties behind them
print(df["Bedrooms"].value_counts().sort_index())
print(df[df["Bedrooms"] >= 18][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].sort_values("Bedrooms", ascending=False).to_string(index=False))
print(df.duplicated().sum())
# Doing the same for bathroom counts
print(df["Bathrooms"].value_counts().sort_index())
print(df[df["Bathrooms"] >= 17][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].sort_values("Bathrooms", ascending=False).to_string(index=False))
# Looking at the largest lots and whether their property types help explain them
print(df["Acreage"].describe().round(2))
print(df.nlargest(20, "Acreage")[[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].to_string(index=False))
print((df["Acreage"] > 100).sum())
print(
    df[df["Acreage"] > 100]["Property Type"].value_counts()
)
print(
    df[df["Acreage"] > 100]["Price"].describe().round(2)
)
print(df["Acreage"].quantile([0.90, 0.95, 0.99, 0.999]))
print(df[df["Acreage"] > 500][[
    "City",
    "Province",
    "Price",
    "Square Footage",
    "Property Type",
    "Bedrooms",
    "Bathrooms",
    "Acreage"
]].sort_values("Acreage", ascending=False).to_string(index=False))
# Checking the upper end of prices and the most expensive listings
print(df["Price"].quantile([0.90, 0.95, 0.99, 0.999]))
print(
    df.nlargest(20, "Price")[
        ["City", "Province", "Price", "Square Footage",
         "Property Type", "Bedrooms", "Bathrooms", "Acreage"]
    ].to_string(index=False)
)
# Looking at coordinate extremes to see which locations need a closer look
print(df["Latitude"].describe().round(2))
print(
    df.nsmallest(10, "Latitude")[
        ["City", "Province", "Latitude", "Longitude", "Price"]
    ].to_string(index=False)
)

print()

print(
    df.nlargest(10, "Latitude")[
        ["City", "Province", "Latitude", "Longitude", "Price"]
    ].to_string(index=False)
)
print(df["Longitude"].describe().round(2))
print(
    df.nsmallest(10, "Longitude")[
        ["City", "Province", "Latitude", "Longitude", "Price"]
    ].to_string(index=False)
)

print()

print(
    df.nlargest(10, "Longitude")[
        ["City", "Province", "Latitude", "Longitude", "Price"]]
)
# Checking how listings are spread across cities and provinces
print(f'Cities, {df["City"].nunique()}')
print(f'Provinces, {df["Province"].nunique()}')
print(df["City"].value_counts().head(20))
print(df["City"].value_counts().tail(20))
print(df["Province"].value_counts())
print(df["Province"].unique())
print(df.groupby("Province")["City"].nunique())
# Checking city names that appear in more than one province
city_province = df.groupby("City")["Province"].nunique()

print(city_province[city_province > 1].tail(51))
# Comparing property types and their price ranges
print(df["Property Type"].value_counts())
print(df["Property Type"].unique())
print(
    df.groupby("Property Type")["Price"]
      .agg(["count", "min", "median", "max"])
      .sort_values("median")
      .round(2)
)
# Looking at how floor area and room counts relate to price
print(df[["Square Footage", "Price"]].corr())
print(df[["Bedrooms", "Price"]].corr())
print(df[["Bathrooms", "Price"]].corr())
# Checking amenity values, keeping missing entries visible
print(df["Garage"].value_counts(dropna=False).to_string())
print(df["Parking"].value_counts(dropna=False).to_string())
# Comparing missing building details across property types
print(df["Basement"].value_counts(dropna=False).to_string())
print(
    df.groupby("Property Type")["Basement"]
      .apply(lambda x: x.isna().sum())
      .sort_values(ascending=False)
      .to_string()
)
print(df["Exterior"].value_counts(dropna=False).to_string())
print(
    df.groupby("Property Type")["Exterior"]
      .apply(lambda x: x.isna().sum())
      .sort_values(ascending=False)
      .to_string()
)
print(df["Flooring"].value_counts(dropna=False).to_string())
print(
    df.groupby("Property Type")["Flooring"]
      .apply(lambda x: x.isna().sum())
      .sort_values(ascending=False)
      .to_string()
)
print(df["Roof"].value_counts(dropna=False).to_string())
print(
    df.groupby("Property Type")["Roof"]
      .apply(lambda x: x.isna().sum())
      .sort_values(ascending=False)
      .to_string()
)
print(df["Heating"].value_counts(dropna=False).to_string())
print(
    df.groupby("Property Type")["Heating"]
      .apply(lambda x: x.isna().sum())
      .sort_values(ascending=False)
      .to_string()
)
# Looking through the remaining amenity categories
print(df["Fireplace"].value_counts(dropna=False).to_string())
print(df["Waterfront"].value_counts(dropna=False).to_string())
print(df["Sewer"].value_counts(dropna=False).to_string())
print(df["Pool"].value_counts(dropna=False).to_string())
print(df["Garden"].value_counts(dropna=False).to_string())
print(df["Balcony"].value_counts(dropna=False).to_string())
# Comparing bedrooms with square footage to see if the values make sense
print(
    df.groupby("Bedrooms")["Square Footage"]
      .agg(["count", "median", "min", "max"])
      .round(2)
      .to_string()
)
# Inspecting small floor areas where the bedroom count seems worth checking
print(
    df.loc[
        (df["Bedrooms"] >= 2) & (df["Square Footage"] < 500),
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Square Footage", "Acreage",
            "Property Type"
        ]
    ].sort_values("Square Footage")
    .to_string(index=False)
)
# Looking more closely at high bedroom counts paired with small floor areas
print(
    df.loc[
        (df["Bedrooms"] >= 8) & (df["Square Footage"] < 1200),
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Square Footage", "Acreage",
            "Property Type"
        ]
    ].sort_values(["Bedrooms", "Square Footage"])
    .to_string(index=False)
)
# Narrowing the small-area check to records below 400 square feet
print(
    df.loc[
        (df["Bedrooms"] >= 2) & (df["Square Footage"] < 400),
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Square Footage", "Acreage",
            "Property Type"
        ]
    ].sort_values("Square Footage")
    .to_string(index=False)
)
# Checking properties with more than twice as many bathrooms as bedrooms
print(
    df.loc[
        df["Bathrooms"] > df["Bedrooms"] * 2,
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Square Footage", "Acreage",
            "Property Type"
        ]
    ]
    .sort_values("Bathrooms", ascending=False)
    .to_string(index=False)
)
# Inspecting zero-bedroom listings that still have several bathrooms
print(
    df.loc[
        (df["Bedrooms"] == 0) & (df["Bathrooms"] >= 3),
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Square Footage", "Acreage",
            "Property Type"
        ]
    ]
    .sort_values(
        ["Bathrooms", "Square Footage"],
        ascending=[False, False]
    )
    .to_string(index=False)
)
# Comparing zero-bedroom listings by property type for context
print(
    df.loc[df["Bedrooms"] == 0]
      .groupby("Property Type")
      .agg(
          count=("Property Type", "size"),
          median_bathrooms=("Bathrooms", "median"),
          median_sqft=("Square Footage", "median"),
          max_bathrooms=("Bathrooms", "max"),
          max_sqft=("Square Footage", "max")
      )
      .sort_values("count", ascending=False)
      .to_string()
)
# Looking specifically at Single Family listings with no recorded bedrooms
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bedrooms"] == 0),
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Square Footage", "Acreage"
        ]
    ]
    .sort_values("Square Footage", ascending=False)
    .to_string(index=False)
)
# Checking listings with neither bedrooms nor bathrooms before drawing conclusions
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bedrooms"] == 0) &
        (df["Bathrooms"] == 0),
        [
            "City", "Province", "Price",
            "Square Footage", "Acreage"
        ]
    ]
    .sort_values("Square Footage", ascending=False)
    .to_string(index=False)
)
# Comparing their provinces, prices, floor areas, and lot sizes
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bedrooms"] == 0) &
        (df["Bathrooms"] == 0)
    ]
    .groupby("Province")
    .agg(
        count=("Price", "size"),
        median_price=("Price", "median"),
        median_sqft=("Square Footage", "median"),
        median_acreage=("Acreage", "median"),
        max_sqft=("Square Footage", "max"),
        max_acreage=("Acreage", "max")
    )
    .sort_values("count", ascending=False)
    .to_string()
)
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bedrooms"] == 0) &
        (df["Bathrooms"] == 0),
        "Square Footage"
    ]
    .describe()
    .round(2)
)
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bedrooms"] == 0) &
        (df["Bathrooms"] == 0),
        "Price"
    ]
    .describe()
    .round(2)
)
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bedrooms"] == 0) &
        (df["Bathrooms"] == 0),
        "Acreage"
    ]
    .describe()
    .round(2)
)
print(df.columns.tolist())
print(df["Basement"].value_counts(dropna=False).to_string())
# Checking for negative measurements and confirming the numeric column types
print(
    df[["Price", "Bedrooms", "Bathrooms", "Acreage", "Square Footage"]]
    .lt(0)
    .sum()
)
print(df[[
    "Latitude",
    "Longitude",
    "Price",
    "Bedrooms",
    "Bathrooms",
    "Acreage",
    "Square Footage"
]].dtypes)
# Revisiting coordinates within provinces rather than only looking at extremes
print(
    df.loc[df["Province"] == "BC", "Longitude"]
      .describe()
      .round(2)
)
print(
    df.groupby("Province")["Longitude"]
      .agg(["count", "min", "median", "max"])
      .round(2)
      .to_string()
)
print(
    df.groupby("Province")["Latitude"]
      .agg(["count", "min", "median", "max"])
      .round(2)
      .to_string()
)
# Checking text values for leading or trailing spaces
categorical_cols = df.select_dtypes(include="str").columns

for col in categorical_cols:
    whitespace_count = df[col].dropna().astype(str).str.strip().ne(
        df[col].dropna().astype(str)
    ).sum()

    print(f"{col}: {whitespace_count}")

print(df["Property Type"].value_counts(dropna=False))
# Looking for city spellings that differ only in capitalization
print(
    df["City"]
      .value_counts()
      .loc[lambda x: x.index.str.lower().duplicated(keep=False)]
)
print(df["Province"].value_counts(dropna=False))
# Comparing garage values across property types
print(
    pd.crosstab(
        df["Property Type"],
        df["Garage"]
    )
)
# Inspecting land listings with garages, then checking their other residential details
print(
    df.loc[
        (df["Property Type"] == "Vacant Land") &
        (df["Garage"] == "Yes"),
        [
            "City", "Province", "Price", "Bedrooms",
            "Bathrooms", "Acreage", "Square Footage",
            "Property Type", "Garage", "Parking"
        ]
    ].to_string(index=False)
)
print(
    df.loc[
        df["Property Type"] == "Vacant Land",
        [
            "City", "Province", "Price",
            "Bedrooms", "Bathrooms",
            "Acreage", "Square Footage",
            "Garage", "Parking",
            "Basement", "Exterior",
            "Fireplace", "Heating",
            "Flooring", "Roof",
            "Waterfront", "Sewer",
            "Pool", "Garden", "Balcony"
        ]
    ].sort_values("Price").to_string(index=False)
)
# Checking expensive listings with small floor areas more closely
print(
    df.loc[
        (df["Price"] >= 5_000_000) & (df["Square Footage"] < 1000),
        [
            "City", "Province", "Price",
            "Bedrooms", "Bathrooms",
            "Square Footage", "Acreage",
            "Property Type"
        ]
    ]
    .sort_values("Price", ascending=False)
    .to_string(index=False)
)
# Pulling up the full Langley record to see the details behind this combination
print(
    df.loc[
        (df["City"] == "Langley") &
        (df["Province"] == "BC") &
        (df["Price"] == 8_990_000) &
        (df["Square Footage"] == 864),
        :
    ].to_string(index=False)
)

# Looking at Single Family listings with bedrooms but no recorded bathrooms
zero_bath_single = df.loc[
    (df["Property Type"] == "Single Family") &
    (df["Bathrooms"] == 0) &
    (df["Bedrooms"] > 0)
]

print("Number of records:", len(zero_bath_single))

print(
    zero_bath_single[
        [
            "City", "Province", "Price",
            "Bedrooms", "Bathrooms",
            "Square Footage", "Acreage"
        ]
    ]
    .sort_values("Square Footage")
    .to_string(index=False)
)

# Checking whether other building details are also missing for these listings
print(
    df.loc[
        (df["Property Type"] == "Single Family") &
        (df["Bathrooms"] == 0) &
        (df["Bedrooms"] > 0),
        ["Bathrooms", "Basement", "Exterior", "Fireplace",
         "Heating", "Flooring", "Roof"]
    ].isna().sum()
)

# Separating smaller and larger homes to put the missing bathrooms in context
print(zero_bath_single["Square Footage"].ge(1000).sum())
print(zero_bath_single["Square Footage"].lt(1000).sum())

print((
    (zero_bath_single["Bedrooms"] >= 2) &
    (zero_bath_single["Square Footage"] >= 1500)
).sum())
print(df[df["Price"] > 100000])
# %% [markdown]
# # PLC Data: Initial Quality Check and Cleaning (Beckhoff)
#
# **Project:** Performance Loss Project
#
# **Script:** `scripts/beckhoff_cleaning.py`
#
# **Input:** `data/raw/Beispieldatensatz Beckhoff SPS.csv` (raw data) and `data/raw/Variablenerklärung.xlsx` (variable documentation)
#
# **Output:** `data/processed/beckhoff_cleaned.csv` (5,197 rows, 54 columns: `timestamp` plus 53 signals)
#
# **Rule:** the raw data is never changed. All cleaning is done on a copy.
#
# Run the cells from top to bottom (Run All), because later cells use variables created in earlier cells.


# %% [markdown]
# ## Step 1 – Setup, loading and first inspection
#
# ### Summary
#
# The project folder is located relative to the script, and the raw CSV is loaded with `;` as separator. The shape, first rows, column names, data types, missing values and duplicate rows are inspected to understand the raw data before changing anything.
#
# ### Code

# %%
from pathlib import Path
import pandas as pd

# Locate the repository root (this script lives in scripts/)
# (__file__ does not exist in the Interactive Window, so fall back to the current folder,
#  which is the folder opened in VS Code)
try:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_DIR = Path.cwd()

# Raw data lives in data/raw/, the cleaned output goes to data/processed/
DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Locate the raw Beckhoff CSV file
csv_path = DATA_DIR / "Beispieldatensatz Beckhoff SPS.csv"
print("Looking for:", csv_path)

# Load the dataset (semicolon-separated)
df = pd.read_csv(
    csv_path,
    sep=";",
    low_memory=False
)

# Inspect the dataset
print("=" * 50)
print("BECKHOFF DATASET INSPECTION")
print("=" * 50)

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nFirst five rows:")
print(df.iloc[:, :6].head())   # only the first 6 columns, so the output stays readable

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values in each column:")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", df.duplicated().sum())

# %% [markdown]
# ### Result and conclusion
#
# - Shape: 5,197 rows × 55 columns (`timestamp`, 53 signals and one extra column `Unnamed: 54`).
# - `Unnamed: 54` has 5,197 missing values, so it is completely empty.
# - `Pallets.Conveyor_belt_lift`, `WH.desiredPalletX` and `WH.desiredPalletY` are integers, all other signals are boolean.
# - Duplicate rows: 0.
#
# The raw file is structurally sound. The only problem is the empty `Unnamed: 54` column, which pandas creates when every line ends with a trailing semicolon. It is handled in Step 2.


# %% [markdown]
# ## Step 2 – Basic data cleaning
#
# ### Summary
#
# A copy of the raw dataframe is made so the original stays untouched. Columns that are completely empty are detected and removed.
#
# ### Code

# %%
# Create a copy so the raw data stays unchanged
clean_df = df.copy()

# Remove columns that are completely empty
empty_columns = clean_df.columns[
    clean_df.isna().all()
].tolist()

print("Completely empty columns:", empty_columns)

clean_df = clean_df.dropna(axis=1, how="all")

print("Original shape:", df.shape)
print("Shape after removing empty columns:", clean_df.shape)

# %% [markdown]
# ### Result and conclusion
#
# - Empty columns found: `Unnamed: 54`.
# - Shape before: (5197, 55). Shape after: (5197, 54).
#
# Only the artefact column was removed. No rows and no real signals were lost, and the raw dataframe `df` can still be used for comparison.


# %% [markdown]
# ## Step 3 – Timestamp analysis
#
# ### Summary
#
# The `timestamp` column (Unix time in milliseconds) is analysed to find out how regularly the PLC data was logged. The difference between consecutive timestamps is calculated and summarised.
#
# ### Code

# %%
timestamp_diff = clean_df["timestamp"].diff()

print("Timestamp information:")
print("First timestamp:", clean_df["timestamp"].iloc[0])
print("Last timestamp:", clean_df["timestamp"].iloc[-1])

print("\nTime difference statistics (milliseconds):")
print(timestamp_diff.describe())

print("\nMost common time differences:")
print(timestamp_diff.value_counts().head(10))

# %% [markdown]
# ### Result and conclusion
#
# - Mean interval: 97.3 ms, median: 93 ms, minimum: 78 ms, maximum: 1,841 ms.
# - Most common intervals: 93 ms (1,912 times), 108 ms (459), 94 ms (410).
#
# The data is logged at roughly 10 Hz, but the sampling is not perfectly regular. The maximum interval of 1,841 ms is far above the typical value and is investigated in Step 4.


# %% [markdown]
# ## Step 4 – Investigate large time gaps
#
# ### Summary
#
# All intervals larger than 200 ms are listed, and the records around the largest gap are displayed. The goal is to decide whether the gaps indicate missing data or just logging jitter.
#
# ### Code

# %%
# Calculate the difference between consecutive timestamps
time_diff = clean_df["timestamp"].diff()

# Show intervals greater than 200 milliseconds
large_gaps = clean_df.loc[
    time_diff > 200,
    ["timestamp"]
].copy()

large_gaps["gap_ms"] = time_diff[time_diff > 200]

print("========== LARGE TIMESTAMP GAPS ==========")
print("Number of gaps above 200 ms:", len(large_gaps))
print(large_gaps.to_string(index=True))

# Display records immediately before and after the largest gap
# (only the first 8 columns, so the output stays readable)
if not large_gaps.empty:
    largest_gap_index = large_gaps["gap_ms"].idxmax()

    print("\nRecords around the largest gap:")
    print(
        clean_df.loc[
            max(0, largest_gap_index - 2):
            largest_gap_index + 2
        ].iloc[:, :8].to_string(index=True)
    )

# %% [markdown]
# ### Result and conclusion
#
# - 33 intervals are larger than 200 ms. The largest is 1,841 ms at row 3217.
# - Around row 3217, several signals change state between the two surrounding rows, so some transitions happened inside the gap and were not logged.
#
# The gaps are a logging characteristic, not missing rows, and only 33 of 5,196 intervals are affected. No rows were removed and nothing was interpolated, because invented values for boolean machine signals would create false events.


# %% [markdown]
# ## Step 5 – Timestamp validation
#
# ### Summary
#
# The millisecond timestamps are converted to readable UTC datetimes to check that they are plausible, to compute the total recording length and to confirm the data is in chronological order.
#
# ### Code

# %%
# Convert milliseconds to readable datetime
clean_df["datetime"] = pd.to_datetime(
    clean_df["timestamp"],
    unit="ms",
    utc=True
)

print("First recorded time:")
print(clean_df["datetime"].iloc[0])

print("\nLast recorded time:")
print(clean_df["datetime"].iloc[-1])

duration_seconds = (
    clean_df["timestamp"].iloc[-1]
    - clean_df["timestamp"].iloc[0]
) / 1000

print("\nTotal recording duration in seconds:")
print(duration_seconds)

print("\nTimestamp order is correct:")
print(clean_df["timestamp"].is_monotonic_increasing)

# %% [markdown]
# ### Result and conclusion
#
# - Recording from 2019-09-26 18:21:14.925 UTC to 18:29:40.319 UTC.
# - Duration: 505.394 seconds (about 8.4 minutes). Timestamps are monotonically increasing.
#
# The timestamps are valid, correctly ordered and describe one continuous recording. The `datetime` column is only a helper and is not saved in the final file.


# %% [markdown]
# ## Step 6 – Inspect the variable explanation file
#
# ### Summary
#
# The Excel file `Variablenerklärung.xlsx` is opened, its sheets are listed and the first rows of each sheet are shown, to understand how the variables are documented.
#
# ### Code

# %%
excel_path = DATA_DIR / "Variablenerklärung.xlsx"

# List the available sheets
excel_file = pd.ExcelFile(excel_path)

print("========== EXCEL SHEET NAMES ==========")
print(excel_file.sheet_names)

# Display the first rows of each sheet
for sheet in excel_file.sheet_names:
    explanation_df = pd.read_excel(
        excel_path,
        sheet_name=sheet
    )

    print(f"\n========== SHEET: {sheet} ==========")
    print("Shape:", explanation_df.shape)
    print(explanation_df.head(10).to_string(index=False))
    print("Column names:", explanation_df.columns.tolist())

# %% [markdown]
# ### Result and conclusion
#
# - Sheets: `Siemens` (142 variables) and `Beckhoff` (53 variables with their station).
#
# The `Beckhoff` sheet is the reference for this dataset. The `Siemens` sheet belongs to the other dataset and is not used here.


# %% [markdown]
# ## Step 7 – Verify the Beckhoff variable mapping
#
# ### Summary
#
# The signal names in the CSV are compared with the variable names in the Excel documentation, in both directions, to make sure every signal is documented and nothing documented is missing from the data.
#
# ### Code

# %%
beckhoff_variables = pd.read_excel(
    excel_path,
    sheet_name="Beckhoff"
)

# Get variable names from the Excel sheet
documented_signals = set(
    beckhoff_variables["Variablenname / Beschreibung"]
    .dropna()
    .astype(str)
    .str.strip()
)

# Get signal names from the cleaned CSV
csv_signals = set(clean_df.columns) - {"timestamp", "datetime"}

# Compare the two lists
missing_from_excel = sorted(csv_signals - documented_signals)
missing_from_csv = sorted(documented_signals - csv_signals)

print("========== BECKHOFF VARIABLE MAPPING ==========")
print("Number of CSV signals:", len(csv_signals))
print("Number of documented signals:", len(documented_signals))

print("\nCSV signals missing from Excel documentation:")
print(missing_from_excel)

print("\nExcel variables missing from the CSV:")
print(missing_from_csv)

print("\nAll CSV signals documented:", len(missing_from_excel) == 0)
print("All documented variables present in CSV:", len(missing_from_csv) == 0)

# %% [markdown]
# ### Result and conclusion
#
# - 53 CSV signals and 53 documented signals. No mismatches in either direction.
#
# The mapping is complete and exact, so no column had to be renamed, added or removed.


# %% [markdown]
# ## Step 8 – Data quality checks
#
# ### Summary
#
# The 53 signal columns are checked for data types, constant signals, number of unique values, missing values and the values of non-boolean signals.
#
# ### Code

# %%
# Exclude timestamp and datetime from signal checks
signal_columns = [
    col for col in clean_df.columns
    if col not in ["timestamp", "datetime"]
]

# 1. Check data types
print("========== SIGNAL DATA TYPES ==========")
print(clean_df[signal_columns].dtypes.to_string())

# %%
# 2. Find columns with only one unique value
constant_columns = [
    col for col in signal_columns
    if clean_df[col].nunique(dropna=False) <= 1
]

print("========== CONSTANT SIGNALS ==========")
print("Number of constant signals:", len(constant_columns))
print(constant_columns)

# %%
# 3. Count unique values in each signal
print("========== UNIQUE VALUE COUNTS ==========")
for col in signal_columns:
    print(f"{col}: {clean_df[col].nunique(dropna=False)}")

# %%
# 4. Check missing values in the cleaned dataset
print("========== MISSING VALUES AFTER CLEANING ==========")
print(clean_df[signal_columns].isna().sum().sort_values(
    ascending=False
).head(10))

# %%
# 5. Check values in non-Boolean signals
print("========== NON-BOOLEAN SIGNAL VALUES ==========")
for col in signal_columns:
    if clean_df[col].dtype != bool:
        print(f"\n{col}")
        print(clean_df[col].value_counts(dropna=False).head(10))

# %% [markdown]
# ### Result and conclusion
#
# - 50 boolean signals and 3 integer signals. Missing values: 0.
# - 8 constant signals (see Step 9).
# - `WH.desiredPalletX` has 4 distinct values and `WH.desiredPalletY` has 3 distinct values.
#
# The signals are complete and consistently typed. The 8 constant signals carry no information for a model, but they are kept in the shared file so the ML stage can decide whether to drop them.


# %% [markdown]
# ## Step 9 – Inspect the constant signal values
#
# ### Summary
#
# The fixed value of each constant signal is printed, so it is known whether the signal is permanently on, permanently off or a fixed number.
#
# ### Code

# %%
print("========== CONSTANT SIGNAL DETAILS ==========")

for col in constant_columns:
    print(f"{col}: {clean_df[col].iloc[0]!r}")

# %% [markdown]
# ### Result and conclusion
#
# - Always False: `VG.I3_ref_switch_rotate`, `WH.I4_Ref_swtich_vertical`, `WH.A1_trail_sensor_lower`, `WH.A2_trail_sensor_upper`, `VG.atHome`, `VG.error_robot`.
# - Always True: `VG.safeCurtainLong`.
# - Always 800: `Pallets.Conveyor_belt_lift`.
#
# These sensors or states never changed during this 8.4-minute recording. Notably, `VG.error_robot` is always False, so error events cannot explain performance loss in this dataset.


# %% [markdown]
# ## Step 10 – Save the cleaned dataset
#
# ### Summary
#
# The final dataset is built from the original `timestamp` and all 53 signals. The helper `datetime` column is excluded, and the result is saved as `beckhoff_cleaned.csv` with final checks.
#
# ### Code

# %%
# Keep the original timestamp and all 53 signal columns.
# Exclude the datetime helper column from the saved dataset.
output_columns = [
    col for col in clean_df.columns
    if col != "datetime"
]

final_df = clean_df[output_columns].copy()

# Save the cleaned dataset in data/processed/
output_path = PROCESSED_DIR / "beckhoff_cleaned.csv"

final_df.to_csv(output_path, index=False)

print("========== FINAL CLEANED DATASET ==========")
print("Saved to:", output_path)
print("Final shape:", final_df.shape)
print("Missing values:", final_df.isna().sum().sum())
print("Duplicate rows:", final_df.duplicated().sum())
print("Timestamp ordered:", final_df["timestamp"].is_monotonic_increasing)
print("Number of signal columns:", len(final_df.columns) - 1)

# %% [markdown]
# ### Result and conclusion
#
# - Final shape: (5197, 54). Missing values: 0. Duplicate rows: 0. Timestamps ordered.
#
# The cleaned Beckhoff dataset is complete, ordered, fully documented and ready to be merged with the Siemens data. Only the empty artefact column was removed and no values were altered.


# %% [markdown]
# ## Step 11 (recommended) – Check for duplicated alias columns
#
# ### Summary
#
# Several output signals appear twice under different names, for example `WH.Q3_horizontal_towards_rack` and `WH.Q3_horRack`. This check shows whether those pairs are identical, so the ML stage does not receive duplicated features. Nothing is dropped here; the result only informs the decision.
#
# ### Code

# %%
# Columns whose values are identical to an earlier column
duplicate_mask = final_df.T.duplicated()
duplicate_columns = final_df.columns[duplicate_mask].tolist()

print("Number of duplicated columns:", len(duplicate_columns))
print(duplicate_columns)

# %% [markdown]
# ### Result and conclusion
#
# Fill in after running: if the pairs are identical, either keep both and state it in the README, or drop one of each pair and document which ones were removed.
# %%

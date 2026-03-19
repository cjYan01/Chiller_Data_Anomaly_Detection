import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import io

st.set_page_config(page_title="Chiller Unit Anomaly Detection Platform", layout="wide")
st.title("🔧 Chiller Unit Data Anomaly Detection Platform")

CHANNELS = [f"CH-{i:02d}" for i in range(1, 10)]
COL_TEMPLATES = {
    "chw_flow_rate": "chw_flow_rate__{}",
    "chw_return_temp": "chw_return_temp__{}",
    "chw_supply_temp": "chw_supply_temp__{}",
    "cdw_flow_rate": "cdw_flow_rate__{}",
    "cdw_return_temp": "cdw_return_temp__{}",
    "cdw_supply_temp": "cdw_supply_temp__{}",
    "cooling_load": "cooling_load__{}",
    "power_consumption": "power_consumption__{}",
    "operation_status": "operation_status__{}",
}

# Physical common-sense rules (for column-level detection)
PHYSICAL_RULES = [
    {"type": "column", "col_pattern": "chw_return_temp", "range": (10, 18), "desc": "Chilled water return temperature 10-18°C"},
    {"type": "column", "col_pattern": "chw_supply_temp", "range": (5, 15), "desc": "Chilled water supply temperature 5-15°C"},
    {"type": "column", "col_pattern": "cdw_return_temp", "range": (30, 35), "desc": "Cooling water return temperature 30-35°C"},
    {"type": "column", "col_pattern": "cdw_supply_temp", "range": (15, 33), "desc": "Cooling water supply temperature 15-33°C"},
    {"type": "column", "col_pattern": "chw_flow_rate", "range": (0, np.inf), "lower_only": False, "desc": "Chilled water flow rate ≥0"},
    {"type": "column", "col_pattern": "cdw_flow_rate", "range": (0, np.inf), "lower_only": False, "desc": "Cooling water flow rate ≥0"},
    {"type": "column", "col_pattern": "power_consumption", "range": (0, np.inf), "lower_only": False, "desc": "Power consumption ≥0"},
    {"type": "delta", "system": "chw", "range": (3, 8), "desc": "Chilled water supply-return temperature difference 3-8°C"},
    {"type": "delta", "system": "cdw", "range": (3, 5), "desc": "Cooling water supply-return temperature difference 3-5°C"},
]

RATE_RULES = [
    {"col_pattern": "chw_return_temp", "rate_range": (-0.5, 0.5), "desc": "Chilled water return temperature ±0.5°C/30min"},
    {"col_pattern": "chw_supply_temp", "rate_range": (-0.5, 0.5), "desc": "Chilled water supply temperature ±0.5°C/30min"},
    {"col_pattern": "cdw_supply_temp", "rate_range": (-1.5, 1.5), "desc": "Cooling water supply temperature ±1.5°C/30min"},
    {"col_pattern": "cdw_return_temp", "rate_range": (-1.5, 1.5), "desc": "Cooling water return temperature ±1.5°C/30min"},
]

WATER_SPECIFIC_HEAT = 4.2
COP_RANGE = (3, 7)
STATUS_ZERO_THRESHOLD = 0.1

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    else:
        st.error("CSV file is missing the 'time' column")
        return None
    return df

# ------------------- Column-level anomaly detection functions -------------------
def detect_physical_threshold_colwise(df, rules, channels):
    """
    Returns a boolean DataFrame of the same shape as df, True indicates an anomaly in that cell.
    Handles single-column rules and delta rules (marks both columns when delta anomaly occurs).
    """
    anomaly_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for rule in rules:
        if rule["type"] == "column":
            col_pattern = rule["col_pattern"]
            lower, upper = rule["range"]
            lower_only = rule.get("lower_only", False)
            for ch in channels:
                col = COL_TEMPLATES[col_pattern].format(ch)
                if col in df.columns:
                    if lower_only:
                        # e.g. >0, anomaly condition is <=0
                        cond = df[col] <= lower
                    else:
                        cond = (df[col] < lower) | (df[col] > upper)
                    anomaly_mask.loc[cond, col] = True
        elif rule["type"] == "delta":
            system = rule["system"]  # 'chw' or 'cdw'
            lower, upper = rule["range"]
            for ch in channels:
                supply_col = COL_TEMPLATES[f"{system}_supply_temp"].format(ch)
                return_col = COL_TEMPLATES[f"{system}_return_temp"].format(ch)
                if supply_col in df.columns and return_col in df.columns:
                    delta = df[return_col] - df[supply_col]
                    cond = (delta < lower) | (delta > upper)
                    # When delta anomaly occurs, mark both supply and return columns as anomalies
                    anomaly_mask.loc[cond, supply_col] = True
                    anomaly_mask.loc[cond, return_col] = True
    return anomaly_mask

def detect_rate_threshold_colwise(df, rules, channels):
    """Rate-of-change detection, returns column-level mask"""
    anomaly_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for rule in rules:
        col_pattern = rule["col_pattern"]
        lower, upper = rule["rate_range"]
        for ch in channels:
            col = COL_TEMPLATES[col_pattern].format(ch)
            if col in df.columns:
                diff = df[col].diff()  # current minus previous
                cond = (diff < lower) | (diff > upper)
                anomaly_mask.loc[cond, col] = True
    return anomaly_mask

def detect_3sigma_rowwise(df, cols):
    """Returns row-level Series (marks entire row)"""
    anomaly = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                cond = (df[col] < mean - 3*std) | (df[col] > mean + 3*std)
                anomaly |= cond
    return anomaly

def detect_iqr_rowwise(df, cols):
    anomaly = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cond = (df[col] < lower) | (df[col] > upper)
            anomaly |= cond
    return anomaly

def detect_rolling_rowwise(df, cols, window_size, threshold_std=3):
    anomaly = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
            cond = (df[col] < rolling_mean - threshold_std * rolling_std) | (df[col] > rolling_mean + threshold_std * rolling_std)
            cond = cond & (rolling_std > 0)
            anomaly |= cond
    return anomaly

def detect_energy_balance_rowwise(df, constant=WATER_SPECIFIC_HEAT, tolerance=0.2):
    # Only checks the four related columns, returns anomaly mask and reference tables
    anomaly = pd.DataFrame(False, index=df.index, columns=[])
    reference_tables = {}
    for ch in CHANNELS:
        flow_col = COL_TEMPLATES["chw_flow_rate"].format(ch)
        return_temp_col = COL_TEMPLATES["chw_return_temp"].format(ch)
        supply_temp_col = COL_TEMPLATES["chw_supply_temp"].format(ch)
        load_col = COL_TEMPLATES["cooling_load"].format(ch)
        # Check if all columns exist
        if all(col in df.columns for col in [flow_col, return_temp_col, supply_temp_col, load_col]):
            delta_t = df[return_temp_col] - df[supply_temp_col]
            theoretical = df[flow_col] * constant * delta_t
            actual = df[load_col]
            lower = theoretical * (1 - tolerance)
            upper = theoretical * (1 + tolerance)
            # Only judge whether cooling_load is within the allowed range
            cond = (actual < lower) | (actual > upper)
            # Output anomaly mask for these four columns
            for col in [flow_col, return_temp_col, supply_temp_col, load_col]:
                if col not in anomaly.columns:
                    anomaly[col] = False
                anomaly.loc[cond, col] = True
            # Reference table
            ref_table = pd.DataFrame({
                "Theoretical load": theoretical,
                "Lower bound": lower,
                "Upper bound": upper,
                "Actual load": actual,
                "Anomaly": cond
            })
            reference_tables[ch] = ref_table
    return anomaly, reference_tables

def detect_cop_anomaly_rowwise(df, cop_range=COP_RANGE):
    anomaly = pd.Series(False, index=df.index)
    for ch in CHANNELS:
        load_col = COL_TEMPLATES["cooling_load"].format(ch)
        power_col = COL_TEMPLATES["power_consumption"].format(ch)
        if load_col in df.columns and power_col in df.columns:
            cop = df[load_col] / (df[power_col] + 1e-9)
            power_zero = df[power_col] == 0
            load_zero = df[load_col] == 0
            cond1 = power_zero & ~load_zero
            cond2 = ~power_zero & ((cop < cop_range[0]) | (cop > cop_range[1]))
            anomaly |= cond1 | cond2
    return anomaly

def detect_status_consistency_rowwise(df, threshold=STATUS_ZERO_THRESHOLD):
    anomaly = pd.Series(False, index=df.index)
    for ch in CHANNELS:
        status_col = COL_TEMPLATES["operation_status"].format(ch)
        flow_cols = [COL_TEMPLATES["chw_flow_rate"].format(ch), COL_TEMPLATES["cdw_flow_rate"].format(ch)]
        power_col = COL_TEMPLATES["power_consumption"].format(ch)
        if status_col in df.columns:
            status_zero = df[status_col] == 0
            for col in flow_cols + [power_col]:
                if col in df.columns:
                    cond = status_zero & (df[col].abs() > threshold)
                    anomaly |= cond
    return anomaly


# Isolation Forest anomaly detection
def detect_isoforest_anomaly_rowwise(df, feature_cols, contamination=0.05, random_state=42):
    data = df[feature_cols].dropna()
    if data.empty:
        return pd.Series(False, index=df.index)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(data)
    # -1 for anomaly, 1 for normal
    anomaly_indices = data.index[preds == -1]
    anomaly = pd.Series(False, index=df.index)
    anomaly[anomaly_indices] = True
    return anomaly

# ------------------- Streamlit UI -------------------
def main():
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file containing the required column names")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    st.sidebar.header("Select Detection Methods")
    methods_to_run = []  # each element: (method_name, params_dict, is_column_level)

    with st.sidebar.expander("🔹 Fixed Threshold Methods", expanded=False):
        use_physical = st.checkbox("Physical range constraints", value=False)
        if use_physical:
            st.write("The following rules will be applied:")
            for rule in PHYSICAL_RULES:
                st.caption(rule["desc"])
            methods_to_run.append(("physical", None, True))  # True means column-level

        use_rate = st.checkbox("Rate-of-change constraints (30 min)", value=False)
        if use_rate:
            st.write("The following rules will be applied:")
            for rule in RATE_RULES:
                st.caption(rule["desc"])
            methods_to_run.append(("rate", None, True))

    with st.sidebar.expander("📊 Statistical Methods", expanded=False):
        use_3sigma = st.checkbox("3-sigma (normal distribution)", value=False)
        if use_3sigma:
            methods_to_run.append(("3sigma", None, False))

        use_iqr = st.checkbox("IQR (box plot)", value=False)
        if use_iqr:
            methods_to_run.append(("iqr", None, False))

        use_rolling = st.checkbox("Rolling window", value=False)
        if use_rolling:
            window = st.number_input("Window size (data points)", min_value=2, value=10, step=1)
            methods_to_run.append(("rolling", {"window": window}, False))

    with st.sidebar.expander("⚙️ Physical Rule Methods", expanded=False):
        use_balance = st.checkbox("Cooling load consistency check", value=False)
        if use_balance:
            constant = st.number_input("Constant (kWh/(m³·°C))", value=WATER_SPECIFIC_HEAT, step=0.01)
            tolerance = st.number_input("Relative error tolerance", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            methods_to_run.append(("energy_balance", {"constant": constant, "tolerance": tolerance}, False))

        use_cop = st.checkbox("COP anomaly detection", value=False)
        if use_cop:
            cop_min = st.number_input("COP minimum", value=float(COP_RANGE[0]), step=0.5)
            cop_max = st.number_input("COP maximum", value=float(COP_RANGE[1]), step=0.5)
            methods_to_run.append(("cop", {"cop_range": (cop_min, cop_max)}, False))

        use_status = st.checkbox("Status consistency check", value=False)
        if use_status:
            status_thresh = st.number_input("Flow/power threshold when status=0", value=STATUS_ZERO_THRESHOLD, step=0.1)
            methods_to_run.append(("status", {"threshold": status_thresh}, False))

    with st.sidebar.expander("🤖 Machine Learning (Not Recommended)", expanded=False):
        use_isoforest = st.checkbox("Isolation Forest anomaly detection", value=False)
        if use_isoforest:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            default_features = [
                "chw_flow_rate__CH-01",
                "cooling_load__CH-01",
                "power_consumption__CH-01",
                "chw_supply_temp__CH-01",
                "chw_return_temp__CH-01"
            ]
            # Keep only columns that actually exist in df
            default_cols = [col for col in default_features if col in numeric_cols]
            selected_features = st.multiselect("Select feature columns", options=numeric_cols, default=default_cols)
            contamination = st.slider("Anomaly ratio", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
            if selected_features:
                methods_to_run.append(("isoforest", {"features": selected_features, "contamination": contamination}, False))

    run_button = st.sidebar.button("🚀 Run Anomaly Detection")


    if run_button:
        if not methods_to_run:
            st.warning("Please select at least one detection method")
            return

        # Initialize total anomaly mask (column-level, all False)
        total_anomaly_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        # Additional: separate masks for physical rules
        physical_column_mask = None
        physical_delta_mask = None

        # Cache for statistical method key parameters
        stat_params = {"3sigma": [], "iqr": [], "rolling": []}

        for method_name, params, is_colwise in methods_to_run:
            st.write(f"Executing: {method_name}")
            if method_name == "physical":
                # Handle the two rule types separately
                column_rules = [r for r in PHYSICAL_RULES if r["type"] == "column"]
                delta_rules = [r for r in PHYSICAL_RULES if r["type"] == "delta"]
                physical_column_mask = detect_physical_threshold_colwise(df, column_rules, CHANNELS)
                physical_delta_mask = detect_physical_threshold_colwise(df, delta_rules, CHANNELS)
                # Merge into total mask
                total_anomaly_mask |= physical_column_mask
                total_anomaly_mask |= physical_delta_mask
            elif method_name == "rate":
                mask = detect_rate_threshold_colwise(df, RATE_RULES, CHANNELS)
                total_anomaly_mask |= mask
            elif method_name == "3sigma":
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col in df.columns:
                        mean = df[col].mean()
                        std = df[col].std()
                        stat_params["3sigma"].append(f"{col}: μ={mean:.4f}, σ={std:.4f}")
                    col_mask = detect_3sigma_rowwise(df, [col])
                    total_anomaly_mask.loc[col_mask, col] = True
            elif method_name == "iqr":
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        stat_params["iqr"].append(f"{col}: Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
                    col_mask = detect_iqr_rowwise(df, [col])
                    total_anomaly_mask.loc[col_mask, col] = True
            elif method_name == "rolling":
                window = params["window"]
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col in df.columns:
                        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                        rolling_std = df[col].rolling(window=window, min_periods=1).std()
                        col_mask = detect_rolling_rowwise(df, [col], window)
                        abn_idx = col_mask[col_mask].index
                        if not abn_idx.empty:
                            param_table = pd.DataFrame({
                                "Anomaly index": abn_idx,
                                "Rolling mean": rolling_mean.loc[abn_idx].values,
                                "Rolling std": rolling_std.loc[abn_idx].values
                            })
                            stat_params["rolling"].append((col, param_table))
                    col_mask = detect_rolling_rowwise(df, [col], window)
                    total_anomaly_mask.loc[col_mask, col] = True
            elif method_name == "energy_balance":
                constant = params["constant"]
                tolerance = params["tolerance"]
                # Only checks four columns, returns anomaly mask and reference tables
                anomaly_mask, ref_tables = detect_energy_balance_rowwise(df, constant, tolerance)
                # Update only the related four columns
                for col in anomaly_mask.columns:
                    total_anomaly_mask[col] = anomaly_mask[col]
                # First display detection results (only the four related columns, anomalies only)
                st.subheader("Detection Results (Only columns involved in cooling load consistency check)")
                # Keep only anomalies, others as NaN
                result_df = df[anomaly_mask.columns].where(anomaly_mask, np.nan)
                rows_with_anomaly = anomaly_mask.any(axis=1)
                display_df = result_df[rows_with_anomaly]
                st.dataframe(display_df)
                # Then display reference tables
                st.subheader("Cooling Load Consistency Check Reference Tables")
                for ch, ref_table in ref_tables.items():
                    st.caption(f"{ch} Cooling load consistency comparison")
                    st.dataframe(ref_table)
            elif method_name == "cop":
                cop_range = params["cop_range"]
                # Compute COP and anomalies per channel
                cop_anomaly_mask = pd.DataFrame(False, index=df.index, columns=[])
                cop_reference_tables = {}
                for ch in CHANNELS:
                    load_col = COL_TEMPLATES["cooling_load"].format(ch)
                    power_col = COL_TEMPLATES["power_consumption"].format(ch)
                    if load_col in df.columns and power_col in df.columns:
                        cop = df[load_col] / (df[power_col] + 1e-9)
                        both_zero = (df[load_col] == 0) & (df[power_col] == 0)
                        cond = ((cop < cop_range[0]) | (cop > cop_range[1])) & (~both_zero)
                        # Output anomaly mask for these two columns
                        for col in [load_col, power_col]:
                            if col not in cop_anomaly_mask.columns:
                                cop_anomaly_mask[col] = False
                            cop_anomaly_mask.loc[cond, col] = True
                        # Reference table
                        ref_table = pd.DataFrame({
                            "Cooling load": df[load_col],
                            "Power": df[power_col],
                            "COP": cop,
                            "Anomaly": cond
                        })
                        cop_reference_tables[ch] = ref_table
                # Update total mask
                for col in cop_anomaly_mask.columns:
                    total_anomaly_mask[col] = cop_anomaly_mask[col]
                # First display detection results (only the two related columns, anomalies only)
                st.subheader("Detection Results (Only columns involved in COP check)")
                result_df = df[cop_anomaly_mask.columns].where(cop_anomaly_mask, np.nan)
                rows_with_anomaly = cop_anomaly_mask.any(axis=1)
                display_df = result_df[rows_with_anomaly]
                st.dataframe(display_df)
                # Then display COP reference tables
                st.subheader("COP Calculation Reference Tables")
                for ch, ref_table in cop_reference_tables.items():
                    st.caption(f"{ch} COP calculation and detection")
                    st.dataframe(ref_table)
            elif method_name == "status":
                threshold = params["threshold"]
                # Check status consistency per channel
                status_anomaly_mask = pd.DataFrame(False, index=df.index, columns=[])
                status_reference_tables = {}
                for ch in CHANNELS:
                    status_col = COL_TEMPLATES["operation_status"].format(ch)
                    flow_cols = [COL_TEMPLATES["chw_flow_rate"].format(ch), COL_TEMPLATES["cdw_flow_rate"].format(ch)]
                    power_col = COL_TEMPLATES["power_consumption"].format(ch)
                    relevant_cols = flow_cols + [power_col]
                    if status_col in df.columns:
                        status_zero = df[status_col] == 0
                        # Check each relevant column
                        for col in relevant_cols:
                            if col in df.columns:
                                cond = status_zero & (df[col].abs() > threshold)
                                if col not in status_anomaly_mask.columns:
                                    status_anomaly_mask[col] = False
                                status_anomaly_mask.loc[cond, col] = True
                        # Reference table
                        ref_table = pd.DataFrame({
                            "Status": df[status_col],
                            flow_cols[0]: df[flow_cols[0]] if flow_cols[0] in df.columns else np.nan,
                            flow_cols[1]: df[flow_cols[1]] if flow_cols[1] in df.columns else np.nan,
                            "Power": df[power_col] if power_col in df.columns else np.nan,
                            "Anomaly_chw_flow": status_anomaly_mask[flow_cols[0]] if flow_cols[0] in status_anomaly_mask.columns else np.nan,
                            "Anomaly_cdw_flow": status_anomaly_mask[flow_cols[1]] if flow_cols[1] in status_anomaly_mask.columns else np.nan,
                            "Anomaly_power": status_anomaly_mask[power_col] if power_col in status_anomaly_mask.columns else np.nan
                        })
                        status_reference_tables[ch] = ref_table
                # Update total mask
                for col in status_anomaly_mask.columns:
                    total_anomaly_mask[col] = status_anomaly_mask[col]
                # First display detection results (only relevant columns, anomalies only)
                st.subheader("Detection Results (Only columns involved in status consistency check)")
                # Ensure operation_status columns show original values always
                result_cols = []
                for ch in CHANNELS:
                    status_col = COL_TEMPLATES["operation_status"].format(ch)
                    chw_col = COL_TEMPLATES["chw_flow_rate"].format(ch)
                    cdw_col = COL_TEMPLATES["cdw_flow_rate"].format(ch)
                    power_col = COL_TEMPLATES["power_consumption"].format(ch)
                    # Only add columns that exist in df
                    group = [c for c in [status_col, chw_col, cdw_col, power_col] if c in df.columns]
                    result_cols.extend(group)
                # Build result table: operation_status columns show original values, others show only where anomaly is True
                result_df = pd.DataFrame(index=df.index)
                for col in result_cols:
                    if col.startswith("operation_status"):
                        result_df[col] = df[col]
                    else:
                        result_df[col] = df[col].where(status_anomaly_mask.get(col, False), np.nan)
                rows_with_anomaly = status_anomaly_mask.any(axis=1)
                display_df = result_df[rows_with_anomaly]
                st.dataframe(display_df)
                # Then display status consistency reference tables
                st.subheader("Status Consistency Check Reference Tables")
                for ch, ref_table in status_reference_tables.items():
                    st.caption(f"{ch} Status consistency check")
                    st.dataframe(ref_table)
            elif method_name == "isoforest":
                features = params["features"]
                contamination = params["contamination"]
                row_mask = detect_isoforest_anomaly_rowwise(df, features, contamination)
                # Display only feature columns, show original values where anomaly is True, else NaN
                st.subheader("Detection Results (Only feature columns with anomalies)")
                result_df = pd.DataFrame(index=df.index)
                for col in features:
                    result_df[col] = df[col].where(row_mask, np.nan)
                display_df = result_df[row_mask]
                st.dataframe(display_df)
                # Update total mask (for potential download etc.)
                for col in features:
                    total_anomaly_mask.loc[row_mask, col] = True

        # Apply mask to generate anomaly result table: keep anomalies, set others to NaN
        result_df = df.where(total_anomaly_mask, other=np.nan).reset_index()

        # Generate cleaned data (anomalies replaced with NaN) —— normal values kept, anomalies set to NaN
        cleaned_df = df.where(~total_anomaly_mask, other=np.nan).reset_index()

        st.subheader("Detection Results")
        # Count total anomalous cells
        anomaly_cell_count = int(total_anomaly_mask.values.sum())
        st.write(f"Total anomalous cells detected: {anomaly_cell_count}")

        # Display results (only rows with anomalies)
        rows_with_anomaly = total_anomaly_mask.any(axis=1)
        display_df = result_df[rows_with_anomaly.values]
        st.dataframe(display_df)

        # Use two columns for download buttons
        col1, col2 = st.columns(2)
        with col1:
            # Download anomaly results CSV (only rows with anomalies)
            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download anomaly results CSV (only anomaly rows)",
                data=csv_buffer.getvalue(),
                file_name="anomaly_results.csv",
                mime="text/csv"
            )
        with col2:
            # Download cleaned data CSV (all rows, anomalies replaced with NaN)
            st.download_button(
                label="📥 Download cleaned data CSV (anomalies replaced with NaN)",
                data=cleaned_df.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        # Output statistical method key parameters after detection
        if any(m[0] == "3sigma" for m in methods_to_run) and stat_params["3sigma"]:
            st.subheader("3-sigma Method Key Parameters")
            for txt in stat_params["3sigma"]:
                st.caption(txt)
        if any(m[0] == "iqr" for m in methods_to_run) and stat_params["iqr"]:
            st.subheader("IQR Method Key Parameters")
            for txt in stat_params["iqr"]:
                st.caption(txt)
        if any(m[0] == "rolling" for m in methods_to_run) and stat_params["rolling"]:
            st.subheader("Rolling Window Method Key Parameters")
            for col, param_table in stat_params["rolling"]:
                st.caption(f"{col} Rolling window anomaly parameters")
                st.dataframe(param_table)

        # Rate-of-change reference tables (if used)
        if any(m[0] == "rate" for m in methods_to_run):
            st.subheader("Rate-of-Change Anomaly Reference Tables")
            # Recompute mask to ensure display
            mask = detect_rate_threshold_colwise(df, RATE_RULES, CHANNELS)
            for rule in RATE_RULES:
                col_pattern = rule["col_pattern"]
                for ch in CHANNELS:
                    col = COL_TEMPLATES[col_pattern].format(ch)
                    if col in df.columns:
                        diff = df[col].diff()
                        abn_idx = mask.index[mask[col]]
                        if not abn_idx.empty:
                            ref_table = pd.DataFrame({
                                "Anomaly value": df.loc[abn_idx, col],
                                "Difference from previous": diff.loc[abn_idx]
                            })
                            st.caption(f"{col} Rate-of-change anomalies")
                            st.dataframe(ref_table)

        # Additional: display the two physical rule masks separately
        if physical_column_mask is not None or physical_delta_mask is not None:
            st.subheader("Physical Rule Anomaly Details")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Single‑column rule anomaly mask**")
                if physical_column_mask is not None:
                    st.dataframe(physical_column_mask)
                else:
                    st.info("No single‑column rule anomalies detected")
            with col2:
                st.markdown("**Delta rule anomaly mask**")
                if physical_delta_mask is not None:
                    st.dataframe(physical_delta_mask)
                else:
                    st.info("No delta rule anomalies detected")

        with st.expander("View detailed anomaly masks per method (boolean)"):
            st.dataframe(total_anomaly_mask)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import io

st.set_page_config(page_title="冷水机组异常检测平台", layout="wide")
st.title("🔧 冷水机组数据异常检测平台")

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

# 物理常识规则（用于列级检测）
PHYSICAL_RULES = [
    {"type": "column", "col_pattern": "chw_return_temp", "range": (10, 18), "desc": "冷冻水回水温度 10-18°C"},
    {"type": "column", "col_pattern": "chw_supply_temp", "range": (5, 15), "desc": "冷冻水供水温度 5-15°C"},
    {"type": "column", "col_pattern": "cdw_return_temp", "range": (30, 35), "desc": "冷却水回水温度 30-35°C"},
    {"type": "column", "col_pattern": "cdw_supply_temp", "range": (15, 33), "desc": "冷却水供水温度 15-33°C"},
    {"type": "column", "col_pattern": "chw_flow_rate", "range": (0, np.inf), "lower_only": False, "desc": "冷冻水流量 ≥0"},
    {"type": "column", "col_pattern": "cdw_flow_rate", "range": (0, np.inf), "lower_only": False, "desc": "冷却水流量 ≥0"},
    {"type": "column", "col_pattern": "power_consumption", "range": (0, np.inf), "lower_only": False, "desc": "功耗 ≥0"},
    {"type": "delta", "system": "chw", "range": (3, 8), "desc": "冷冻水供回水温差 3-8°C"},
    {"type": "delta", "system": "cdw", "range": (3, 5), "desc": "冷却水供回水温差 3-5°C"},
]

RATE_RULES = [
    {"col_pattern": "chw_return_temp", "rate_range": (-0.5, 0.5), "desc": "冷冻水回水温度 ±0.5°C/30min"},
    {"col_pattern": "chw_supply_temp", "rate_range": (-0.5, 0.5), "desc": "冷冻水供水温度 ±0.5°C/30min"},
    {"col_pattern": "cdw_supply_temp", "rate_range": (-1.5, 1.5), "desc": "冷却水供水温度 ±1.5°C/30min"},
    {"col_pattern": "cdw_return_temp", "rate_range": (-1.5, 1.5), "desc": "冷却水回水温度 ±1.5°C/30min"},
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
        st.error("CSV文件中缺少'time'列")
        return None
    return df

# ------------------- 列级异常检测函数 -------------------
def detect_physical_threshold_colwise(df, rules, channels):
    """
    返回与df形状相同的布尔DataFrame，True表示该单元格异常。
    处理单列规则和温差规则（温差异常时标记两列）。
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
                        # 例如 >0，异常条件是 <=0
                        cond = df[col] <= lower
                    else:
                        cond = (df[col] < lower) | (df[col] > upper)
                    anomaly_mask.loc[cond, col] = True
        elif rule["type"] == "delta":
            system = rule["system"]  # 'chw' 或 'cdw'
            lower, upper = rule["range"]
            for ch in channels:
                supply_col = COL_TEMPLATES[f"{system}_supply_temp"].format(ch)
                return_col = COL_TEMPLATES[f"{system}_return_temp"].format(ch)
                if supply_col in df.columns and return_col in df.columns:
                    delta = df[return_col] - df[supply_col]
                    cond = (delta < lower) | (delta > upper)
                    # 温差异常时，将供水和回水两列都标记为异常
                    anomaly_mask.loc[cond, supply_col] = True
                    anomaly_mask.loc[cond, return_col] = True
    return anomaly_mask

def detect_rate_threshold_colwise(df, rules, channels):
    """变化率检测，返回列级掩码"""
    anomaly_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for rule in rules:
        col_pattern = rule["col_pattern"]
        lower, upper = rule["rate_range"]
        for ch in channels:
            col = COL_TEMPLATES[col_pattern].format(ch)
            if col in df.columns:
                diff = df[col].diff()  # 后减前
                cond = (diff < lower) | (diff > upper)
                anomaly_mask.loc[cond, col] = True
    return anomaly_mask

def detect_3sigma_rowwise(df, cols):
    """返回行级Series（整行异常标记）"""
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
    # 只检测相关四列，返回异常掩码和参考表格
    anomaly = pd.DataFrame(False, index=df.index, columns=[])
    reference_tables = {}
    for ch in CHANNELS:
        flow_col = COL_TEMPLATES["chw_flow_rate"].format(ch)
        return_temp_col = COL_TEMPLATES["chw_return_temp"].format(ch)
        supply_temp_col = COL_TEMPLATES["chw_supply_temp"].format(ch)
        load_col = COL_TEMPLATES["cooling_load"].format(ch)
        # 检查所有列都存在
        if all(col in df.columns for col in [flow_col, return_temp_col, supply_temp_col, load_col]):
            delta_t = df[return_temp_col] - df[supply_temp_col]
            theoretical = df[flow_col] * constant * delta_t
            actual = df[load_col]
            lower = theoretical * (1 - tolerance)
            upper = theoretical * (1 + tolerance)
            # 只判断cooling_load是否在允许范围内
            cond = (actual < lower) | (actual > upper)
            # 只在这四列输出异常掩码
            for col in [flow_col, return_temp_col, supply_temp_col, load_col]:
                if col not in anomaly.columns:
                    anomaly[col] = False
                anomaly.loc[cond, col] = True
            # 参考表格
            ref_table = pd.DataFrame({
                "理论冷量": theoretical,
                "允许下界": lower,
                "允许上界": upper,
                "实际冷量": actual,
                "是否异常": cond
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


# Isolation Forest 异常检测
def detect_isoforest_anomaly_rowwise(df, feature_cols, contamination=0.05, random_state=42):
    data = df[feature_cols].dropna()
    if data.empty:
        return pd.Series(False, index=df.index)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    preds = model.fit_predict(data)
    # -1为异常，1为正常
    anomaly_indices = data.index[preds == -1]
    anomaly = pd.Series(False, index=df.index)
    anomaly[anomaly_indices] = True
    return anomaly

# ------------------- Streamlit界面 -------------------
def main():
    uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
    if uploaded_file is None:
        st.info("请上传包含指定列名的CSV文件")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    st.subheader("数据预览")
    st.dataframe(df.head(10))

    st.sidebar.header("检测方法选择")
    methods_to_run = []  # 每个元素为 (方法名, 参数字典, 是否列级)

    with st.sidebar.expander("🔹 固定阈值方法", expanded=False):
        use_physical = st.checkbox("物理常识范围设定", value=False)
        if use_physical:
            st.write("将应用以下规则：")
            for rule in PHYSICAL_RULES:
                st.caption(rule["desc"])
            methods_to_run.append(("physical", None, True))  # True表示列级

        use_rate = st.checkbox("变化率范围设定 (30分钟)", value=False)
        if use_rate:
            st.write("将应用以下规则：")
            for rule in RATE_RULES:
                st.caption(rule["desc"])
            methods_to_run.append(("rate", None, True))

    with st.sidebar.expander("📊 统计异常方法", expanded=False):
        use_3sigma = st.checkbox("正态分布3σ", value=False)
        if use_3sigma:
            methods_to_run.append(("3sigma", None, False))

        use_iqr = st.checkbox("箱线图法 (IQR)", value=False)
        if use_iqr:
            methods_to_run.append(("iqr", None, False))

        use_rolling = st.checkbox("滑动窗口法", value=False)
        if use_rolling:
            window = st.number_input("窗口大小 (数据点数)", min_value=2, value=10, step=1)
            methods_to_run.append(("rolling", {"window": window}, False))

    with st.sidebar.expander("⚙️ 物理规则方法", expanded=False):
        use_balance = st.checkbox("冷量一致性检查", value=False)
        if use_balance:
            constant = st.number_input("常数 (kWh/(m³·°C))", value=WATER_SPECIFIC_HEAT, step=0.01)
            tolerance = st.number_input("相对误差容忍度", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            methods_to_run.append(("energy_balance", {"constant": constant, "tolerance": tolerance}, False))

        use_cop = st.checkbox("能效异常检测 (COP)", value=False)
        if use_cop:
            # 保证value和step类型一致（都为float）
            cop_min = st.number_input("COP最小值", value=float(COP_RANGE[0]), step=0.5)
            cop_max = st.number_input("COP最大值", value=float(COP_RANGE[1]), step=0.5)
            methods_to_run.append(("cop", {"cop_range": (cop_min, cop_max)}, False))

        use_status = st.checkbox("状态一致性检查", value=False)
        if use_status:
            status_thresh = st.number_input("状态为0时流量/功率阈值", value=STATUS_ZERO_THRESHOLD, step=0.1)
            methods_to_run.append(("status", {"threshold": status_thresh}, False))

    with st.sidebar.expander("🤖 机器学习方法（不适合）", expanded=False):
        use_isoforest = st.checkbox("Isolation Forest异常检测", value=False)
        if use_isoforest:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            default_features = [
                "chw_flow_rate__CH-01",
                "cooling_load__CH-01",
                "power_consumption__CH-01",
                "chw_supply_temp__CH-01",
                "chw_return_temp__CH-01"
            ]
            # 只保留实际存在于df的列
            default_cols = [col for col in default_features if col in numeric_cols]
            selected_features = st.multiselect("选择特征列", options=numeric_cols, default=default_cols)
            contamination = st.slider("异常比例", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
            if selected_features:
                methods_to_run.append(("isoforest", {"features": selected_features, "contamination": contamination}, False))

    run_button = st.sidebar.button("🚀 执行异常检测")


    if run_button:
        if not methods_to_run:
            st.warning("请至少选择一种检测方法")
            return

        # 初始化总异常掩码（列级，全False）
        total_anomaly_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        # 新增：物理常识规则分为两类掩码
        physical_column_mask = None
        physical_delta_mask = None

        # 统计异常方法关键参数缓存
        stat_params = {"3sigma": [], "iqr": [], "rolling": []}

        for method_name, params, is_colwise in methods_to_run:
            st.write(f"正在执行: {method_name}")
            if method_name == "physical":
                # 分别处理两类规则
                column_rules = [r for r in PHYSICAL_RULES if r["type"] == "column"]
                delta_rules = [r for r in PHYSICAL_RULES if r["type"] == "delta"]
                physical_column_mask = detect_physical_threshold_colwise(df, column_rules, CHANNELS)
                physical_delta_mask = detect_physical_threshold_colwise(df, delta_rules, CHANNELS)
                # 合并到总掩码
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
                                "异常点": abn_idx,
                                "窗口均值": rolling_mean.loc[abn_idx].values,
                                "窗口标准差": rolling_std.loc[abn_idx].values
                            })
                            stat_params["rolling"].append((col, param_table))
                    col_mask = detect_rolling_rowwise(df, [col], window)
                    total_anomaly_mask.loc[col_mask, col] = True
            elif method_name == "energy_balance":
                constant = params["constant"]
                tolerance = params["tolerance"]
                # 只检测四列，返回异常掩码和参考表
                anomaly_mask, ref_tables = detect_energy_balance_rowwise(df, constant, tolerance)
                # 只更新相关四列
                for col in anomaly_mask.columns:
                    total_anomaly_mask[col] = anomaly_mask[col]
                # 先显示检测结果（只显示相关四列的异常）
                st.subheader("检测结果(仅包含冷却量一致性检查相关列)")
                # 只保留异常值，其余为NaN
                result_df = df[anomaly_mask.columns].where(anomaly_mask, np.nan)
                rows_with_anomaly = anomaly_mask.any(axis=1)
                display_df = result_df[rows_with_anomaly]
                st.dataframe(display_df)
                # 再显示参考表格
                st.subheader("冷量一致性检查参考表")
                for ch, ref_table in ref_tables.items():
                    st.caption(f"{ch} 冷量一致性对比")
                    st.dataframe(ref_table)
            elif method_name == "cop":
                cop_range = params["cop_range"]
                # 针对每个通道分别计算COP及异常
                cop_anomaly_mask = pd.DataFrame(False, index=df.index, columns=[])
                cop_reference_tables = {}
                for ch in CHANNELS:
                    load_col = COL_TEMPLATES["cooling_load"].format(ch)
                    power_col = COL_TEMPLATES["power_consumption"].format(ch)
                    if load_col in df.columns and power_col in df.columns:
                        cop = df[load_col] / (df[power_col] + 1e-9)
                        both_zero = (df[load_col] == 0) & (df[power_col] == 0)
                        cond = ((cop < cop_range[0]) | (cop > cop_range[1])) & (~both_zero)
                        # 只在这两列输出异常掩码
                        for col in [load_col, power_col]:
                            if col not in cop_anomaly_mask.columns:
                                cop_anomaly_mask[col] = False
                            cop_anomaly_mask.loc[cond, col] = True
                        # 参考表格
                        ref_table = pd.DataFrame({
                            "冷量": df[load_col],
                            "功耗": df[power_col],
                            "COP": cop,
                            "是否异常": cond
                        })
                        cop_reference_tables[ch] = ref_table
                # 更新总掩码
                for col in cop_anomaly_mask.columns:
                    total_anomaly_mask[col] = cop_anomaly_mask[col]
                # 先显示检测结果（只显示相关两列的异常，非异常为NaN）
                st.subheader("检测结果(仅包含冷却量一致性检查相关列)")
                result_df = df[cop_anomaly_mask.columns].where(cop_anomaly_mask, np.nan)
                rows_with_anomaly = cop_anomaly_mask.any(axis=1)
                display_df = result_df[rows_with_anomaly]
                st.dataframe(display_df)
                # 再显示COP计算结果参考表
                st.subheader("COP计算结果参考表")
                for ch, ref_table in cop_reference_tables.items():
                    st.caption(f"{ch} COP计算与检测")
                    st.dataframe(ref_table)
            elif method_name == "status":
                threshold = params["threshold"]
                # 针对每个通道分别检查状态一致性
                status_anomaly_mask = pd.DataFrame(False, index=df.index, columns=[])
                status_reference_tables = {}
                for ch in CHANNELS:
                    status_col = COL_TEMPLATES["operation_status"].format(ch)
                    flow_cols = [COL_TEMPLATES["chw_flow_rate"].format(ch), COL_TEMPLATES["cdw_flow_rate"].format(ch)]
                    power_col = COL_TEMPLATES["power_consumption"].format(ch)
                    relevant_cols = flow_cols + [power_col]
                    if status_col in df.columns:
                        status_zero = df[status_col] == 0
                        # 检查每个相关列
                        for col in relevant_cols:
                            if col in df.columns:
                                cond = status_zero & (df[col].abs() > threshold)
                                if col not in status_anomaly_mask.columns:
                                    status_anomaly_mask[col] = False
                                status_anomaly_mask.loc[cond, col] = True
                        # 参考表格
                        ref_table = pd.DataFrame({
                            "状态": df[status_col],
                            flow_cols[0]: df[flow_cols[0]] if flow_cols[0] in df.columns else np.nan,
                            flow_cols[1]: df[flow_cols[1]] if flow_cols[1] in df.columns else np.nan,
                            "功耗": df[power_col] if power_col in df.columns else np.nan,
                            "是否异常_冷冻水流量": status_anomaly_mask[flow_cols[0]] if flow_cols[0] in status_anomaly_mask.columns else np.nan,
                            "是否异常_冷却水流量": status_anomaly_mask[flow_cols[1]] if flow_cols[1] in status_anomaly_mask.columns else np.nan,
                            "是否异常_功耗": status_anomaly_mask[power_col] if power_col in status_anomaly_mask.columns else np.nan
                        })
                        status_reference_tables[ch] = ref_table
                # 更新总掩码
                for col in status_anomaly_mask.columns:
                    total_anomaly_mask[col] = status_anomaly_mask[col]
                # 先显示检测结果（只显示相关列的异常，非异常为NaN）
                st.subheader("检测结果(仅包含状态一致性检查相关列)")
                # 保证包含operation_status_CH-0X列，且其值始终显示原值
                # 整理每个通道相关列依次排列
                result_cols = []
                for ch in CHANNELS:
                    status_col = COL_TEMPLATES["operation_status"].format(ch)
                    chw_col = COL_TEMPLATES["chw_flow_rate"].format(ch)
                    cdw_col = COL_TEMPLATES["cdw_flow_rate"].format(ch)
                    power_col = COL_TEMPLATES["power_consumption"].format(ch)
                    # 只添加存在于df的列
                    group = [c for c in [status_col, chw_col, cdw_col, power_col] if c in df.columns]
                    result_cols.extend(group)
                # 构造结果表：operation_status列直接显示原值，其余列异常显示原值，非异常为NaN
                result_df = pd.DataFrame(index=df.index)
                for col in result_cols:
                    if col.startswith("operation_status"):
                        result_df[col] = df[col]
                    else:
                        result_df[col] = df[col].where(status_anomaly_mask.get(col, False), np.nan)
                rows_with_anomaly = status_anomaly_mask.any(axis=1)
                display_df = result_df[rows_with_anomaly]
                st.dataframe(display_df)
                # 再显示状态一致性检查参考表
                st.subheader("状态一致性检查参考表")
                for ch, ref_table in status_reference_tables.items():
                    st.caption(f"{ch} 状态一致性检测")
                    st.dataframe(ref_table)
            elif method_name == "isoforest":
                features = params["features"]
                contamination = params["contamination"]
                row_mask = detect_isoforest_anomaly_rowwise(df, features, contamination)
                # 只显示特征列，异常显示原值，非异常为NaN
                st.subheader("检测结果(仅包含特征列异常值)")
                result_df = pd.DataFrame(index=df.index)
                for col in features:
                    result_df[col] = df[col].where(row_mask, np.nan)
                display_df = result_df[row_mask]
                st.dataframe(display_df)
                # 更新总掩码（如需后续下载等操作）
                for col in features:
                    total_anomaly_mask.loc[row_mask, col] = True

        # 应用掩码生成异常结果表：保留异常值，其余置NaN
        result_df = df.where(total_anomaly_mask, other=np.nan).reset_index()


        st.subheader("检测结果")
        # 统计异常单元格总数
        anomaly_cell_count = int(total_anomaly_mask.values.sum())
        st.write(f"共检测到 {anomaly_cell_count} 个异常单元格")

        # 显示结果（只显示有异常的行）
        rows_with_anomaly = total_anomaly_mask.any(axis=1)
        display_df = result_df[rows_with_anomaly.values]
        st.dataframe(display_df)


        # 下载按钮
        csv_buffer = io.StringIO()
        display_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 下载异常结果CSV",
            data=csv_buffer.getvalue(),
            file_name="anomaly_results.csv",
            mime="text/csv"
        )

        # 检测结果后输出统计异常方法关键参数
        if any(m[0] == "3sigma" for m in methods_to_run) and stat_params["3sigma"]:
            st.subheader("3σ方法关键参数")
            for txt in stat_params["3sigma"]:
                st.caption(txt)
        if any(m[0] == "iqr" for m in methods_to_run) and stat_params["iqr"]:
            st.subheader("IQR方法关键参数")
            for txt in stat_params["iqr"]:
                st.caption(txt)
        if any(m[0] == "rolling" for m in methods_to_run) and stat_params["rolling"]:
            st.subheader("滑动窗口法关键参数")
            for col, param_table in stat_params["rolling"]:
                st.caption(f"{col} 滑动窗口异常参数")
                st.dataframe(param_table)

        # 变化率异常参考表格（如有）
        if any(m[0] == "rate" for m in methods_to_run):
            st.subheader("变化率异常参考表")
            # 重新计算mask，保证展示
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
                                "异常值": df.loc[abn_idx, col],
                                "与前一时刻差值": diff.loc[abn_idx]
                            })
                            st.caption(f"{col} 变化率异常")
                            st.dataframe(ref_table)

        # 新增：分别展示物理常识规则两类掩码
        if physical_column_mask is not None or physical_delta_mask is not None:
            st.subheader("物理常识规则异常识别详情")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**单列规则异常掩码**")
                if physical_column_mask is not None:
                    st.dataframe(physical_column_mask)
                else:
                    st.info("未检测单列规则异常")
            with col2:
                st.markdown("**温差规则异常掩码**")
                if physical_delta_mask is not None:
                    st.dataframe(physical_delta_mask)
                else:
                    st.info("未检测温差规则异常")

        with st.expander("查看各方法检测详情（布尔掩码）"):
            st.dataframe(total_anomaly_mask)

if __name__ == "__main__":
    main()
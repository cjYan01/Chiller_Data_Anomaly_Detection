# Chiller_Data_Anomaly_Detection

This project is an interactive data anomaly detection tool built with Streamlit, specifically designed for analyzing operational data from chiller plants. Users can upload CSV files containing the required columns, select from various detection methods (fixed thresholds, statistical methods, physical rules, machine learning), identify anomalies, and download the results or cleaned data.

## Data Schema

The uploaded CSV file must contain the following columns (the time column must be named `time`; other columns follow the naming convention below):

- **time**: Timestamp of the record, recommended format `YYYY-MM-DD HH:MM:SS`.
- **Chilled Water System (CHW)**  
  - `chw_flow_rate_CH-01` ~ `chw_flow_rate_CH-09`: Chilled water flow rate for each chiller  
  - `chw_return_temp_CH-01` ~ `chw_return_temp_CH-09`: Chilled water return temperature (typically higher than supply temperature)  
  - `chw_supply_temp_CH-01` ~ `chw_supply_temp_CH-09`: Chilled water supply temperature (usually 6–12°C)
- **Condenser Water System (CDW)**  
  - `cdw_flow_rate_CH-01` ~ `cdw_flow_rate_CH-09`: Condenser water flow rate  
  - `cdw_return_temp_CH-01` ~ `cdw_return_temp_CH-09`: Condenser water return temperature  
  - `cdw_supply_temp_CH-01` ~ `cdw_supply_temp_CH-09`: Condenser water supply temperature (normally higher than return temperature)
- **Cooling Load**  
  - `cooling_load_CH-01` ~ `cooling_load_CH-09`: Cooling load provided by each chiller
- **Power Consumption**  
  - `power_consumption_CH-01` ~ `power_consumption_CH-09`: Power consumption of each chiller
- **Operation Status**  
  - `operation_status_CH-01` ~ `operation_status_CH-09`: 0 indicates off, non‑zero (usually 1) indicates running

> Note: CH-01 through CH-09 represent up to 9 chillers. You may include only the columns that exist in your data; missing columns are simply ignored.

## Anomaly Detection Methods

The platform offers multiple detection methods that can be combined freely via the sidebar. All detected anomalies are merged into a single boolean mask. The final output includes an anomaly results table (showing original values where anomalies occur) and a cleaned dataset (anomalies replaced with NaN).

### 1. Fixed Threshold Methods

#### 1.1 Physical Range Constraints
- `chw_return_temp_CH-0X`: 10–18°C  
- `chw_supply_temp_CH-0X`: 5–15°C  
- `chw_return_temp - chw_supply_temp` (chilled water delta T): 3–8°C  
- `cdw_return_temp_CH-0X`: 30–35°C  
- `cdw_supply_temp_CH-0X`: 15–33°C  
- `cdw_return_temp - cdw_supply_temp` (condenser water delta T): 3–5°C  
- `chw_flow_rate_CH-0X`: ≥0 (anomaly if ≤0)  
- `cdw_flow_rate_CH-0X`: ≥0  
- `power_consumption_CH-0X`: ≥0  

#### 1.2 Rate‑of‑Change Limits (30‑minute difference)
- `chw_return_temp_CH-0X`: ±0.5°C  
- `chw_supply_temp_CH-0X`: ±0.5°C  
- `cdw_supply_temp_CH-0X`: ±1.0°C to ±1.5°C  
- `cdw_return_temp_CH-0X`: ±1.0°C to ±1.5°C  

### 2. Statistical Methods

#### 2.1 3‑Sigma (Normal Distribution)
For each numeric column, calculate the mean μ and standard deviation σ. A data point is considered anomalous if `|x - μ| > 3σ`.

#### 2.2 IQR (Box Plot)
Compute the first quartile Q1, third quartile Q3, and interquartile range IQR = Q3 − Q1. Anomalies are defined as:
- `x < Q1 − 1.5 × IQR`
- `x > Q3 + 1.5 × IQR`

#### 2.3 Rolling Window
A rolling window (adjustable size) moves along the time series, calculating the local mean and standard deviation. A point is flagged as anomalous if its deviation from the rolling mean exceeds `threshold × rolling_std` (default threshold = 3) and the rolling standard deviation is positive. This method adapts to gradual changes in the data.

### 3. Physical Rule Methods

#### 3.1 Cooling Load Consistency Check
Based on the thermodynamic relation, the theoretical cooling load should approximately equal flow × specific heat × temperature difference:
theoretical_load = chw_flow_rate × WATER_SPECIFIC_HEAT × ΔT
where:
- `ΔT = chw_return_temp − chw_supply_temp`
- `WATER_SPECIFIC_HEAT` defaults to 4.2 (adjustable)

The actual cooling load (`cooling_load`) must lie within a tolerance band around the theoretical value:
ower_bound = theoretical_load × (1 − tolerance)
upper_bound = theoretical_load × (1 + tolerance)

If the actual load is outside this range, all four columns involved (`chw_flow_rate`, `chw_return_temp`, `chw_supply_temp`, `cooling_load`) are marked as anomalous.

#### 3.2 COP (Coefficient of Performance) Anomaly
`COP = cooling_load / power_consumption` (a small epsilon is added to the denominator to avoid division by zero). Anomaly conditions:
- If `power_consumption` is zero but `cooling_load` is not zero → anomaly (energy used without cooling)
- If both are non‑zero and COP falls outside the defined range (default 3–7) → anomaly

The COP range can be adjusted in the UI.

#### 3.3 Status Consistency Check
When `operation_status` is 0 (indicating the chiller is off), the corresponding chilled water flow, condenser water flow, and power consumption should be near zero. If any of these values exceed a threshold (default 0.1) in absolute value, they are flagged as anomalous.

### 4. Machine Learning Method

#### Isolation Forest
An unsupervised anomaly detection algorithm suitable for high‑dimensional data. Users can select which feature columns to include and set the expected contamination ratio (default 0.05). Rows predicted as anomalies by the model will have all selected feature columns marked as anomalous in the final mask.

## Usage

### Install Dependencies
Ensure Python 3.8+ and install the required libraries:
```bash
pip install streamlit pandas numpy scikit-learn

streamlit run detect.py

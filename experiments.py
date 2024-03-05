import mlflow
from mlflow.models import infer_signature

import pandas as pd
from datetime import datetime

from src.prepocessing import preprocessing_df, get_x_y, shifting
from sklearn.preprocessing import StandardScaler
from src.cv import TimeSeriesSlicingCV
from src.utils import get_dynamic_test_set_splitter
from sklearn.feature_selection import RFECV
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


xgb_model = xgb.XGBRegressor() # model selection 

## Setting on the start_date
PROVIDED_START_DATE = datetime.strptime("2020-01-04", '%Y-%m-%d')
PROVIDED_END_DATE = datetime.today().strftime('%Y-%m-%d')
generated_date_range = pd.date_range(start=PROVIDED_START_DATE, end=PROVIDED_END_DATE, freq='W-Mon')

## Reading the input and target series file
sample_source_file = # Provided souce file path
n_row_start = 3
df = pd.read_excel(sample_source_file, skiprows= n_row_start)
mapper = pd.read_excel(sample_source_file, sheet_name="mapping")
mapper = mapper.rename(columns={"series_id": "Input Series"})
mapper['Input Series'] = mapper['Input Series'].astype(int)

## Repplace the columns
df_testing = df.copy()
df_testing.columns = df_testing.columns.astype(str)
df_clean = df_testing.rename(columns=lambda x: x.split(' (')[0])
df_clean = df_clean.rename(columns={"Series ID":"date"})

## Getting the latest input date index
df_date = df_clean.copy()["date"].sort_values(ascending=False)
latest_date = df_date.max()

## Replacing the datetime indexing
df_completed_date = pd.date_range(start=PROVIDED_START_DATE, end=latest_date, freq='W-Mon')
df_testing = df_clean.set_index("date", drop=True)
df_testing = df_testing.reindex(df_completed_date)

## Checking the available date or range that all columns is have values
isna_value_count_by_cols = df_testing.isna().sum()

##################### FEATURES ENGINEERING ############################
## Removing the series with not exceed the threshold
series_to_be_drop = isna_value_count_by_cols[isna_value_count_by_cols >= 100]
dropped_series_list = series_to_be_drop.index.to_list() # Can be the df to store the LIST and have the issue on this
dropped_series_df = pd.DataFrame(series_to_be_drop, columns=['missing_timepoint_count'])
dropped_series_df = dropped_series_df.reset_index(drop=False, names="Input Series")
dropped_series_df['Input Series'] = dropped_series_df['Input Series'].astype(int) 
print(f"Timepoint that exceed the threshold {dropped_series_list}")
cleaned_df = df_testing.drop(columns=dropped_series_list)

## detect the latest timepoint threashold legitable ### optinal
isna_count_by_row = cleaned_df.isna().sum(axis=1)

# target_input split()
target_series, input_series = get_x_y(cleaned_df)

# Shifting 
shifted_input_df = input_series.shift(periods=1, freq='W-Mon')

# OOS mask split
X_train, X_test, y_train= preprocessing_df(shifted_input_df, target_series)

# INPUT series update
interpolated_X_train = X_train.interpolate(method="linear", limit_direction="forward")
interpolated_y_train = y_train.interpolate(method="linear", limit_direction="forward")
interpolated_X_test = X_test.interpolate(method="linear", limit_direction="backward")

# CV
cv_config = {
    "initial_window": 40,
    "horizon": 6,
    "fixed_window": True,
    "block_size": 1,
}
cv = TimeSeriesSlicingCV(**cv_config)


# Define model and hyperparameters
model_param = {}
model = xgb.XGBRegressor(**model_param)

split_structur= get_dynamic_test_set_splitter(cv,interpolated_X_train, interpolated_y_train)
y_prediction = []
mse_records = []
mape_records = []
for window_id, (train_index, test_index) in split_structur:
    X_train = interpolated_X_train.iloc[train_index]
    y_train = interpolated_y_train.iloc[train_index]
    X_test = interpolated_X_train.iloc[test_index]
    y_test = interpolated_y_train.iloc[test_index]

    model_processor = make_pipeline(StandardScaler(), xgb_model)
    model_processor.fit(X_train, y_train)

    prediction = model_processor.predict(X_test)
    y_pred = pd.Series(prediction, index=X_test.index)
    y_prediction.append(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    mse_records.append(pd.Series(mse, index=y_test.index[0]))
    mape_records.append(pd.Series(mse, index=y_test.index[0]))

#######################################################################################
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("us_retail_sales")

# Start an MLflow run
with mlflow.start_run(run_name="LEO"):
    # Log the hyperparameters
    mlflow.log_params(model_param)

    # Log the loss metric
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("MAPE", mape)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "XGB.Regressor for US Retail Sales")

    # Infer the model signature
    signature = infer_signature(interpolated_X_train, model.predict(interpolated_X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="us_retail_sales_items",
        signature=signature,
        input_example=interpolated_X_train,
        registered_model_name="US_RETAIL_SALES",
    )




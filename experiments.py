import mlflow

 mlflow.set_descripttion("Dropped Columns selection")

def train_and_evaluation(df)
  model.fit(X_train, y_train)
  # Evaluate the model
  y_pred = model.predict(X_test)

err = mean_squared_error(y_test, y_pred)
mlflow.log_metric("MSE", err)

# Testing columns to drop to know which cols are the best to drop
columns_to_drop = feature_columns + [None]

for to_drop in columns_to_drop:
  if to_drop:
    dropped = selected.drop([to_drop], axis=1)
  else:
    dropped = selected

  with mlflow.start_run():
    mlflow.log_param("dropped_column", to_drop)
    prepared = prepared_data(dropped)
    train_and_evaluation(prepared)

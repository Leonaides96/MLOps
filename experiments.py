import os
import pickle
import mlflow

mlflow.set_descripttion("Dropped Columns selection")

os.makedirs("tmp")

def train_and_evaluation(df)
  # Train test split the dataset
  ...

  # Save the dataset to mlflow
  dataset_data.to_csv("tmp/dataset.csv", index=False)
  
  # Save the plotted graph to mlflow
  plot = dataframe.plot.scatter(x=0, y="salesprice")
  fig = plot.get_figure()
  fig.savefig("tmp/plot.png")

  # Model training
  model.fit(X_train, y_train)
  
  # Save the model
  serialized_model = pickle.dumps(model)
  with open("tmp/model.pkl", "wb") as f:
    f.write(serialized_model)
  
  # using the artifile to store model registry from the local to endpoint, can adding the path level if necessary
  mlflow.log_artifact("tmp/model.pkl") # store a file from the path
  mlflow.log_artifacts("tmp") # store all the files contain from the parent path

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


# The end to remove the tmp tree level (optional)
shutil.rmtree("tmp")
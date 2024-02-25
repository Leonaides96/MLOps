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
  

  # Creating and store the model information to artifact, or using the customize model
  mlflow.sklearn.log_model(model, "model_name") # Origin
  mlflow.pyfunc.log_model( 
    "custom_model",
    python_model= WrappedLRModel(sklearn_features=list(feature.columns), cat_features=list(),
    artifact=[],
  )# Customized model
  

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


############### Custom Model ############### 
from mlflow.pyfunc import PythonModel

# Creating the class to have the inheritance the model to customize of

class WrappedLRModel(PythonModel):

  def __init__(self, sklearn_features, cat_features)
    """
    cat_features: mapping from categories features name to all possible value
    eg:
    {
    "Bldg type":["1fam", "som", ... ]
    
    } 

    """
    self.features_name = sklearn_features
    self.cat_features = cat_features

  def load_context(self, context):
    with open(context.artifacts['orignal_sklearn_model'], "rb") as r:
      self.lr_model = pickle.load(r) # Example

  def _encode(self, row, col):
    value = row[col]
    row[value] = 1 
    return row

  def predict(self, context, model_input):
    model_features = model_input
    for col, unique_values in self.cat_features.items():
      for uv in unique_values:
        model_features[uv] = 0
      model_features = model_features.apply(lambda x: self._encode(x, col), axis=1)
    model_features = model_features.loc[:, self.features_name]


    return self.lr_model.predict(model_features.to_numpy())
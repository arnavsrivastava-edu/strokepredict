import h2o
from h2o.automl import H2OAutoML
h2o.init()
data = h2o.import_file("stroke_data.csv")
train, test = data.split_frame(ratios=[0.8], seed=42)
target = "stroke"
features = data.columns
features.remove(target)
aml = H2OAutoML(max_models=10, seed=42)
aml.train(x=features, y=target, training_frame=train)
lb = aml.leaderboard
print(lb)
performance = aml.leader.model_performance(test)
print(performance)
predictions = aml.leader.predict(test)
print(predictions)
model_path = h2o.save_model(model=aml.leader, path="model", force=True)
print(f"Model saved to: {model_path}")
# import mlflow
# import pandas as pd

# # Connect to local database
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# # Get all experiments 
# exps = mlflow.search_experiments()
# exp_ids = [e.experiment_id for e in exps]

# # Search for the champion across all of them
# runs = mlflow.search_runs(experiment_ids=exp_ids, order_by=["metrics.r2 DESC"])


# if not runs.empty:
#     best_run = runs.iloc[0]
#     parent_experiment = mlflow.get_experiment(best_run['experiment_id'])
#     print(f"🏆 Champion Found!")
#     print(f"Folder (Experiment): {parent_experiment.name}")
#     print(f"R2 Score: {best_run['metrics.r2']:.4f}")
#     print(f"Run ID: {best_run['run_id']}")
# else:
#     print("No runs found in the database.")
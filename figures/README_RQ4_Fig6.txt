RQ4_Fig6.pdf - SHAP Feature Importance

STATUS: To be generated

DESCRIPTION:
This figure shows SHAP (SHapley Additive exPlanations) based global feature 
importance for the Random Forest model. It provides interpretable insights into
which features have the most impact on academic performance predictions.

TO GENERATE THIS FIGURE:
Run the script: python generate_rq4_fig6.py

OR run the complete pipeline and then call:
  from src.evaluation.visualizations import plot_rq4_fig6_shap_importance
  from src.modeling.train import load_model, load_abt, prepare_data, split_data
  
  rf_model = load_model('models/rf_pass_prediction.pkl')
  abt = load_abt()
  X, y = prepare_data(abt)
  X_train, X_test, y_train, y_test = split_data(X, y)
  plot_rq4_fig6_shap_importance(rf_model, X_test)

The figure will be saved as: figures/RQ4_Fig6.pdf

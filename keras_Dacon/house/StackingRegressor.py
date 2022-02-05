# from sklearn.ensemble import StackingRegressor

# stack_models = [ ('elasticnet', poly_pipeline), 
#                 ('randomforest', rfr), ('gbr', gbr), 
#                 ('lgbm', lgbm) ] 

# stack_reg = StackingRegressor(stack_models, final_estimator=xgb, n_jobs=-1) 

# stack_reg.fit(x_train, y_train) stack_pred = stack_reg.predict(x_test)

# #Weighted Blending
# final_outputs = { 'elasticnet' : poly_pred, 'randomforest' : rfr_pred, 'gbr' : gbr_pred, 'xgb' : xgb_pred, 'lgbm' : lgbm_pred, 'stacking' : stack_pred }

# final_prediction= final_outputs['elasticnet'] * 0.1 +final_outputs['randomforest'] * 0.1 +final_outputs['gbr'] * 0.2  +final_outputs['xgb'] * 0.2 +final_outputs['lgbm'] * 0.2 +final_outputs['stacking'] * 0.2



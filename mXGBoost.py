import pandas as pd
import gc
from xgboost import XGBRegressor

test = pd.read_csv(r'dataset\test.csv')
data = pd.read_pickle(r'dataset\traintest.pkl')

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect()

# =============================================================================
# the lines where we make our predictions
# =============================================================================
model = XGBRegressor(
    max_depth=8,    
    n_estimators=50,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42,
    n_jobs=-1)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train),(X_valid,Y_valid)],
    verbose=True, 
    early_stopping_rounds = 2)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(r'outputdata\XGBoost.csv', index=False)
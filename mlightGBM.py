import pandas as pd
import gc
import lightgbm as lgb

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
model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=32,
    colsample_bytree=0.9497036,
    subsample=0.8715623,
    max_depth=8,
    reg_alpha=0.04,
    reg_lambda=0.073,
    min_split_gain=0.0222415,
    min_child_weight=40)
model.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], early_stopping_rounds=2)

Y_test = model.predict(X_test).clip(0, 20)

    
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(r'outputdata\LGBm.csv', index=False)
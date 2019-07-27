import pandas as pd
import gc

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
from sklearn import tree
model = tree.DecisionTreeRegressor()
model = model.fit(X_train, Y_train)

Y_test = model.predict(X_test).clip(0, 20)

    
submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv(r'outputdata/DecisionTree.csv', index=False)
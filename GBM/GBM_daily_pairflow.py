import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


''' data '''
# first download our CSV file from zenodo
# import dataset
df = pd.read_csv("train_daily_pairflow_exp_8-4.csv")
# print(len(df))
X = df.iloc[:, 1: ]
y = df.iloc[:, 0]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# parameters
params = {
    'objective': 'reg:squarederror',  # Regression task
    'max_depth': 12,                   # Maximum depth of trees
    'learning_rate': 0.1,             # Learning rate
    'n_estimators': 100,              # Number of trees
    'subsample': 0.8,                 # Proportion of data used to train each tree
    'colsample_bytree': 1,            # Proportion of features used to train each tree
    'alpha': 0.1                      # Weight of L1 regularization term
}

''' train '''
evals_result = {}
model = xgb.train(params, dtrain, num_boost_round=800, evals=[(dtrain, 'train'), (dtest, 'test')],
                  evals_result=evals_result, verbose_eval=False)

''' predict '''
y_pred_train = model.predict(dtrain)
y_pred_test = model.predict(dtest)
# R2_score
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print("R2 Score on training data: {:.3f}".format(r2_train))
print("R2 Score on testing data: {:.3f}".format(r2_test))
# RMSE,MAE
train_rmse = evals_result['train']['rmse'][-1]
test_rmse = evals_result['test']['rmse'][-1]
train_mae = np.mean(np.abs(y_train - y_pred_train))
test_mae = np.mean(np.abs(y_test - y_pred_test))



''' SHAP '''
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test, check_additivity=False)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('/data/yuting/code/fig/XGBoost/SHAP_summary_plot_daily_pairflow_exp_8-4.png')


''' scatter plot '''
plt.figure(figsize=(4, 4))
plt.scatter(y_test, y_pred_test, alpha=0.5, color='#B2B6C1')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('true daily pairflow', fontproperties=my_font)
plt.ylabel('predict daily pairflow', fontproperties=my_font)

plt.text(0.05, 0.92, f'$R^2: {r2_test:.2f}$', transform=plt.gca().transAxes, fontproperties=my_font)
plt.text(0.05, 0.85, f'RMSE: {float(test_rmse):.2f}', transform=plt.gca().transAxes, fontproperties=my_font, )
plt.text(0.05, 0.78, f'MAE: {float(test_mae):.2f}', transform=plt.gca().transAxes,fontproperties=my_font, )
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)
plt.tight_layout()
plt.savefig('/data/yuting/code/fig/XGBoost/scatter_daily_pairflow_exp_8-4.png',dpi=300)
plt.show()


import os

# data
dat_dir = os.path.join('.','data')
data_path = os.path.join(dat_dir, 'raw/kc_house_data.csv')
data_path_raw = os.path.join(dat_dir, 'raw/kc_house_data.csv')
data_path_clean = os.path.join(dat_dir, 'processed/data_cleaned_encoded.csv')

data_path_train = os.path.join(dat_dir, 'raw/train.csv')
data_path_test = os.path.join(dat_dir, 'raw/test.csv')
compression = None

report_dir = os.path.join('.','reports')
path_report_pandas_profiling = os.path.join(report_dir,'report_pandas_profiling.html')
path_report_sweetviz = os.path.join(report_dir,'report_sweetviz.html')

# params
model_type = 'regression'
target = 'price'
train_size = 0.8
test_size = 1-train_size
SEED = 100

# model dumps
model_dump_cb = './models/model_cb.dump'

# data Processing
# NOTE: do not do log transform of target here, do that in modelling
cols_log_small = ['sqft_living',
                'sqft_living15',
                'sqft_lot',
                'sqft_lot15'
                ]

cols_drop_small = ['id','date']

cols_sq = ['bedrooms','bathrooms',
        'floors','waterfront','view',
        'age','age_after_renovation',
        'log1p_sqft_living','log1p_sqft_lot',
        'log1p_sqft_above','log1p_sqft_basement',
        'log1p_sqft_living15','log1p_sqft_lot15']

cols_drop = ['id', 'date', 'zipcode_top10']

# params
# NOTE: for small model with only raw features, params_rf_small gives better rmse
params_rf_small = dict(n_estimators= 155,random_state=SEED,
    max_features='auto',max_depth=23,min_samples_split=10)

params_rf = dict(n_estimators=1200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=5,
                max_features=None,
                oob_score=True,
                n_jobs=-1,
                random_state=SEED)

params_hgbr = dict(
    l2_regularization=0.0,
    learning_rate=0.01, # default 0.1
    loss='least_squares',
    max_bins=255,
    max_depth=None,
    max_iter=5000, # default 100
    max_leaf_nodes=31,
    min_samples_leaf=20,
    n_iter_no_change=10,
    random_state=SEED,
    early_stopping=True, # default None
    scoring=None,
    tol=1e-07,
    validation_fraction=0.1,
    verbose=0,
    warm_start=False
    )

params_xgb = dict(n_jobs=-1,
                random_state=SEED,
                objective='reg:squarederror',
                n_estimators=1200,
                max_depth=3,  # default 6
                reg_alpha=1,  # default alpha = 0,  alias reg_alpha
                reg_lambda=5, # default lambda = 1, alias reg_lambda
                subsample=1,  # default 1
                gamma=0, # default gamma=0 alias min_split_loss
                min_child_weight=1, # default 1
                colsample_bytree=1, # default 1
                learning_rate=0.1,   # default eta = 0.3
                tree_method = 'auto', # default auto, use gpu_hist
                )

params_lgb = {
    'boosting_type': 'gbdt', # default gbdt
    'colsample_bytree': 0.67211, # default 1.0
    'learning_rate': 0.02169, # default 0.1
    'max_depth': 15, # default -1
    'n_estimators': 750, # default 100
    'num_leaves': 38, # default 31
    'reg_lambda': 0.604 # default 0.0
            }

params_cb ={
'loss_function': 'RMSE',
'random_state': 0, # 0 gives better result
'verbose': False
}

lst_cat_features = ['bedrooms','waterfront','view','condition','grade','zipcode']
# params_cb['cat_features'] = lst_cat_features
# idx_cat_features = [list(df_Xtrain.columns).index(i) for i in lst_cat_features]


# Default Parameters
default_params = """
xgboost parameters:
https://xgboost.readthedocs.io/en/latest/parameter.html

https://www.analyticsvidhya.com/blog/2016/03/
complete-guide-parameter-tuning-xgboost-with-codes-python/

objective
---------
reg:squarederror reg:gamma reg:tweedie
binary:logistic
count:poisson
multi:softmax (we must use num_class parameter)
multi:softprob (softprob outputs vector of ndata*nclass)


================================================= xgboost
XGBRegressor(base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=1,
    gamma=0,
    importance_type='gain',
    learning_rate=0.1,
    max_delta_step=0,
    max_depth=3,
    min_child_weight=1,
    missing=None,
    n_estimators=100,
    n_jobs=1,
    nthread=None,
    objective='reg:linear',
    random_state=0,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    seed=None,
    silent=None,
    subsample=1,
    verbosity=1)


================================================= lightgbm
lightgbm.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=- 1,
    learning_rate=0.1,
    n_estimators=100,
    subsample_for_bin=200000,
    objective=None,
    class_weight=None,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None,
    n_jobs=- 1,
    silent=True,
    importance_type='split',
    **kwargs)

#========================================== catboost
https://catboost.ai/docs/concepts/python-reference_catboostregressor.html

CatBoostRegressor(
iterations=None,
learning_rate=None,
loss_function='RMSE',
use_best_model=None,
verbose=None,
silent=None,
logging_level=None,
one_hot_max_size=None,
ignored_features=None,
train_dir=None,
custom_metric=None,
eval_metric=None,
subsample=None,
max_depth=None,
n_estimators=None,
num_boost_round=None,
num_trees=None,
colsample_bylevel=None,
random_state=None,
reg_lambda=None,
objective=None,
eta=None,
max_bin=None,
early_stopping_rounds=None,
cat_features=None,
min_child_samples=None,
max_leaves=None,
num_leaves=None,
score_function=None,
)
"""
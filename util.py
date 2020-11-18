import numpy as np
import pandas as pd

import sklearn
from sklearn import metrics as skmetrics
import config

def show_methods(obj, ncols=7,start=None, inside=None):
    """ Show all the attributes of a given method.
    Example:
    ========
    show_method_attributes(list)
    """

    print(f'Object Type: {type(obj)}\n')
    lst = [elem for elem in dir(obj) if elem[0]!='_' ]
    lst = [elem for elem in lst
            if elem not in 'os np pd sys time psycopg2'.split() ]

    if isinstance(start,str):
        lst = [elem for elem in lst if elem.startswith(start)]

    if isinstance(start,tuple) or isinstance(start,list):
        lst = [elem for elem in lst for start_elem in start
                if elem.startswith(start_elem)]

    if isinstance(inside,str):
        lst = [elem for elem in lst if inside in elem]

    if isinstance(inside,tuple) or isinstance(inside,list):
        lst = [elem for elem in lst for inside_elem in inside
                if inside_elem in elem]

    return pd.DataFrame(np.array_split(lst,ncols)).T.fillna('')

def print_time_taken(time_taken):
    h,m = divmod(time_taken,60*60)
    m,s = divmod(m,60)
    time_taken = f"{h:.0f} h {m:.0f} min {s:.2f} sec" if h > 0 else f"{m:.0f} min {s:.2f} sec"
    time_taken = f"{m:.0f} min {s:.2f} sec" if m > 0 else f"{s:.2f} sec"

    print(f"\nTime Taken: {time_taken}")

def adjustedR2(rsquared,nrows,ncols):
    return rsquared- (ncols-1)/(nrows-ncols) * (1-rsquared)

def print_regr_eval(ytest,ypreds,ncols,print_=True):
    ytest = np.array(ytest).flatten()
    ypreds = np.array(ypreds).flatten()
    rmse = np.sqrt(skmetrics.mean_squared_error(ytest,ypreds))
    r2 = skmetrics.r2_score(ytest,ypreds)
    ar2 = adjustedR2(r2,len(ytest),ncols)
    evs = skmetrics.explained_variance_score(ytest, ypreds)
    p = ncols      # num of predictors
    N = len(ytest) # sample size

    out = """
Explained Variance: {:.6f}\n
         R-Squared: {:,.6f}\n
             RMSE : {:,.2f}\n
Adjusted R-squared: {:,.6f}
""".format(evs,r2,rmse,ar2)

    warn = """
WARNING: Here adjusted R-squared value is greater than 1.
Please check for number of samples and number of predictors.
    p = {} >= N = {}
""".format(p,N)

    if ar2 > 1.0:
        out = out + '\n' + warn

    if print_:
        print('ytest :', ytest[:3])
        print('ypreds:', ypreds[:3])
        print(out)
    return out

def write_regr_eval(ytest,ypreds,ncols,ofile):
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(ytest,ypreds))
    r2 = sklearn.metrics.r2_score(ytest,ypreds)
    ar2 = adjustedR2(r2,len(ytest),ncols)
    evs = sklearn.metrics.explained_variance_score(ytest, ypreds)

    df_out = pd.DataFrame(
        {'Explained Variance': [evs],
         'R-Squared'         : [r2],
         'RMSE'              : [rmse],
         'Adjusted R-squared': [ar2]
        })

    df_out.to_csv(ofile, index=False)

def clean_data(df,log=True,sq=True,logsq=True,dummy=True,dummy_cat=False):

    df = df.copy()

    # Date time features
    df['date'] = pd.to_datetime(df['date'])
    df['yr_sales'] = df['date'].dt.year
    df['age'] = df['yr_sales'] - df['yr_built']
    df['yr_renovated2'] = np.where(df['yr_renovated'].eq(0), df['yr_built'], df['yr_renovated'])
    df['age_after_renovation'] = df['yr_sales'] - df['yr_renovated2']

    # Boolean data types
    f = lambda x: 1 if x>0 else 0
    df['basement_bool'] = df['sqft_basement'].apply(f)
    df['renovation_bool'] = df['yr_renovated'].apply(f)

    # Numerical features binning
    cols_bin = ['age','age_after_renovation']
    df['age_cat'] = pd.cut(df['age'], 10, labels=range(10)).astype(str)
    df['age_after_renovation_cat'] = pd.cut(df['age_after_renovation'],
                                            10, labels=range(10))

    # Log transformation of large numerical values
    cols_log = ['sqft_living', 'sqft_lot', 'sqft_above',
                'sqft_basement', 'sqft_living15', 'sqft_lot15']
    if log:
        for col in cols_log:
            df['log1p_' + col] = np.log1p(df[col])

    # squared columns
    cols_sq = [
        # cats
        'bedrooms','bathrooms','floors','waterfront','view',

        # created nums
        'age','age_after_renovation']

    if sq:
        for col in cols_sq:
            df[col + '_sq'] = df[col]**2

    cols_log_sq = [
        # log nums
        'log1p_sqft_living','log1p_sqft_lot',
        'log1p_sqft_above','log1p_sqft_basement',
        'log1p_sqft_living15','log1p_sqft_lot15'
        ]
    if logsq:
        for col in cols_log_sq:
            df[col + '_sq'] = df[col]**2

    # Categorical Features
    cols_dummy     = ['waterfront', 'view', 'condition', 'grade']
    cols_dummy_cat = ['age_cat', 'age_after_renovation_cat']
    for c in cols_dummy:
        df[c] = df[c].astype(str)

    # Create dummy variables
    if dummy:
        df_dummy = pd.get_dummies(df[cols_dummy],drop_first=False)
        df       = pd.concat([df,df_dummy], axis=1)

    # dummy variable for newly created cats from numerical feature
    if dummy_cat:
        df_dummy = pd.get_dummies(df[cols_dummy_cat],drop_first=False)
        df       = pd.concat([df,cols_dummy_cat], axis=1)

    # after creating dummy, make the columns number
    for c in cols_dummy + cols_dummy_cat:
        df[c] = df[c].astype(np.int32)

    # Drop unwanted columns
    cols_drop = ['id','date']
    df = df.drop(cols_drop,axis=1)

    return df

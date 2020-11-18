# Imports
import numpy as np
import pandas as pd
import os
import time

# internet
import urllib
import codecs
import base64

# special Imports
import shap
import pandas_profiling
import sweetviz as sv
import streamlit as st
import streamlit.components.v1 as stc
from streamlit_pandas_profiling import st_profile_report

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# local imports
import config
import util
from util import clean_data
from util import print_regr_eval

# modelling
import catboost as cb

# settings
st.set_option("deprecation.showfileUploaderEncoding", False)

__doc__ = """
Date   : Nov 16, 2020
Author : Bhishan Poudel
Purpose: Interactive report of the project

Command:
streamlit run app.py

"""

# Parameters
data_path_train = config.data_path_train
data_path_test = config.data_path_test
target = config.target
logtarget = True

params_data = dict(log=True, sq=True, logsq=False, dummy=True, dummy_cat=False)
params_cb = config.params_cb
model_dump_cb = config.model_dump_cb

path_report_pandas_profiling = config.path_report_pandas_profiling
path_report_sweetviz = config.path_report_sweetviz

# decorator function for logging
def st_log(func):
    def log_func(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time() - start
        st.text("Log: the function `%s` tooks %0.4f seconds" % (func.__name__, end))
        return res

    return log_func

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    stc.html(shap_html, height=height)

def get_table_download_link(df, filename='data.csv', linkname='Download Data'):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linkname}</a>'
    return href

@st.cache
def load_data(data_path, nrows=None):
    data = pd.read_csv(data_path, nrows=nrows)
    return data

def home():
    """Main function """

    # Project Title
    html = """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;"> King County House Price Prediction</p></div>
	"""
    st.markdown(html, unsafe_allow_html=True)

    # Author
    html = """<marquee style='width: 30%; color: blue;'><b> Author: Bhishan Poudel</b></marquee>"""
    st.markdown(html, unsafe_allow_html=True)

    # Load the data
    df_train = load_data(data_path_train)
    df_test_raw = load_data(data_path_test)

    # Download test sample
    df_test_sample = df_test_raw.sample(10)
    if st.checkbox("Download Sample Test data"):
        st.write(get_table_download_link(df_test_sample),
                unsafe_allow_html=True)

    # Upload a file
    if st.checkbox("Upload Your Test data (else full data is used.)"):
        uploaded_file_buffer = st.file_uploader("")
        df_test = pd.read_csv(uploaded_file_buffer)
        df_test = df_test.head(100)
        st.text(f"Shape of Test Data: {df_test.shape}")
        st.dataframe(df_test.head())
    else:
        df_test = df_test_raw
        st.text(f"Using data: {data_path_test}")
        st.text(f"Shape of Test Data: {df_test.shape}")
        st.dataframe(df_test.head())

    # Test Data Description
    st.header("Test Data Description")

    # Show Column Names
    if st.checkbox("Show Columns Names Before Data Cleaning"):
        st.write(df_test.columns.tolist())

    # Data Cleaning and feature selection
    df_train = clean_data(df_train, **params_data)
    df_test = clean_data(df_test, **params_data)
    features = list(sorted(df_train.columns.drop(target)))
    features = [i for i in features if i in df_test.columns]

    df_Xtrain = df_train[features]
    df_Xtest  = df_test[features]
    ytrain = np.array(df_train[target]).flatten()
    if logtarget:
        ytrain = np.log1p(ytrain)

    # Show Column Names
    if st.checkbox("Show Columns Names After Data Cleaning"):
        st.write(features)

    # Model Prediction
    st.header("Model Prediction")

    # if train and test do not have same features, we need to train again
    if list(sorted(df_train.columns)) != list(sorted(df_test.columns)):
        st.text("Model is training, Please wait ...")
        model = cb.CatBoostRegressor(**params_cb)
        model.fit(df_Xtrain,ytrain)
        if os.path.exists(model_dump_cb):
            os.remove(model_dump_cb)
        model.save_model(model_dump_cb)
    else:
        model = cb.CatBoostRegressor()
        model = model.load_model(model_dump_cb)

    ypreds = model.predict(df_test[features])
    ypreds = np.array(ypreds).flatten()
    if logtarget:
        ypreds = np.expm1(ypreds)

    return (model,df_test,ypreds,features)

def model_evaluation(model,df_test,ypreds,features):
    df_Xtest = df_test[features]
    df_out = pd.DataFrame()
    df_out[target] = df_test[target]
    df_out["predicted_" + target] = ypreds

    st.text("Model prediction for first 10 rows:")
    st.text(f"Shape of Test Data: {df_test.shape}")
    df_out_style = df_out.head(10).style.format("{:,.0f}")
    st.dataframe(df_out_style)

    out = print_regr_eval(df_test[target], ypreds, len(features), print_=False)
    st.text(out)

    # Model evaluation
    st.header("Model Evaluation")
    st.subheader("Feature Importance")
    df_fimp = model.get_feature_importance(prettified=True)[['Importances','Feature Id']]
    df_fimp_style = df_fimp.style.background_gradient(subset=['Importances'])
    st.dataframe(df_fimp_style)

def model_evaluation_shap(model,df_test,features):
    df_Xtest = df_test[features]

    # shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_Xtest)
    expected_value = explainer.expected_value

    st.subheader("Summary plot of test data")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_Xtest)
    st.pyplot(fig)

    st.subheader("Summary plot (barplot) of test data")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df_Xtest, plot_type='bar')
    st.pyplot(fig)
    plt.gca()

    st.subheader("Dependence plot between two features")
    feature_x = st.selectbox("feature_x e.g. sqft_living", features)
    feature_y = st.selectbox("feature_y e.g. grade", features)
    st.text("You selected feature_x = {} and feature_y = {}".format(feature_x, feature_y))
    fig, ax = plt.subplots()
    shap.dependence_plot(ind=feature_x, interaction_index=feature_y,
                    shap_values=shap_values,
                    features=df_Xtest,
                    ax=ax)
    st.pyplot(fig)

    # https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html
    st.subheader("Dependence plot for Nth rank feature")
    st.text("Note: 0 is the most importance feature not 1.\nThe y-axis feature is selected automatically.")
    rank_n = st.slider("rank_n", 0, len(features)-1,0)
    fig, ax = plt.subplots()
    shap.dependence_plot(ind="rank("+str(rank_n)+")",
                    shap_values=shap_values,
                    features=df_Xtest,
                    ax=ax)
    st.pyplot(fig)

def get_pandas_profile(df):
    profile = pandas_profiling.ProfileReport(df)
    return profile

def st_display_html(path_html,width=1000,height=500):
    fh_report = codecs.open(path_html,'r')
    page = fh_report.read()
    stc.html(page,height=height,width=width,scrolling=True)

def get_pandas_profile_st():
    st.subheader("Automated EDA with Pandas Profile")
    height = st.slider("height", 400, 1000,800)
    width = st.slider("width", 400, 1000,800)
    st_display_html(path_report_pandas_profiling,
                        width=width,height=height)

def get_sweetviz_profile_st():
    st.subheader("Automated EDA with Sweetviz")
    height = st.slider("height", 400, 1000,800)
    width = st.slider("width", 400, 1000,800)
    st_display_html(path_report_sweetviz,
                        width=width,height=height)

if __name__ == "__main__":
    menu = ["Home","Pandas Profile", "Sweetviz Profile","About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        model,df_test,ypreds,features = home()
        if target not in df_test.columns:
            st.header("Model predictions")
            st.write(ypreds)
        else:
            model_evaluation(model,df_test,ypreds,features)
            model_evaluation_shap(model,df_test,features)
    elif choice == "Pandas Profile":
        get_pandas_profile_st()
    elif choice == "Sweetviz Profile":
        get_sweetviz_profile_st()
    else:
        page = codecs.open('about.md','r').read()
        st.markdown(page)


# %pylab inline

import os, glob, sys, time, datetime
import pandas as pd
import pandas.io.sql as pdsql
from pandas import DataFrame, Series
import math
import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array
import scipy as sp
from scipy import stats
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import dates
import matplotlib.font_manager as font_manager
import seaborn as sns

#맑은고딕체
sns.set(style="whitegrid", font="Malgun Gothic", font_scale=1.5)
matplotlib.rcParams['figure.figsize'] = [20, 10]
fp = font_manager.FontProperties(fname="C:\\WINDOWS\\Fonts\\malgun.TTF", size=15)

#나눔고딕체
# sns.set(style="whitegrid", font="NanumGothic", font_scale=1.5)
# matplotlib.rcParams['figure.figsize'] = [20, 10]
# fp = font_manager.FontProperties(fname="C:\\WINDOWS\\Fonts\\NanumGothic.TTF", size=15)

#새굴림체
# sns.set(style="whitegrid", font="New Gulim", font_scale=1.5)
# matplotlib.rcParams['figure.figsize'] = [20, 10]
# fp = font_manager.FontProperties(fname="C:\\WINDOWS\\Fonts\\NGULIM.TTF", size=15)

def comma_volume(x, pos):  # formatter function takes tick label and tick position
    s = '{:0,d}K'.format(int(x/1000))
    return s

def comma_price(x, pos):  # formatter function takes tick label and tick position
    s = '{:0,d}'.format(int(x))
    return s

def comma_percent(x, pos):  # formatter function takes tick label and tick position
    s = '{:+.2f}'.format(x)
    return s

major_date_formatter = dates.DateFormatter('%Y-%m-%d')
minor_date_formatter = dates.DateFormatter('%m')
price_formatter = ticker.FuncFormatter(comma_price)
volume_formatter = ticker.FuncFormatter(comma_volume)
percent_formatter = ticker.FuncFormatter(comma_percent)

sns.set(style="whitegrid", font="Malgun Gothic", font_scale=1.5)
matplotlib.rcParams['figure.figsize'] = [20, 10]
fp = font_manager.FontProperties(fname="C:\\WINDOWS\\Fonts\\malgun.TTF", size=15)

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier, XGBModel
from xgboost import plot_importance, plot_tree

def XGBoost_Feature(model, df, X_columns, y_columns, performCV=True, printFeatureImportance=True, cv_folds=5):
    print(model)

    model.fit(df[X_columns], df[y_columns])
        
    df_predictions = model.predict(df[X_columns])
    try:
        df_predprob = model.predict_proba(df[X_columns])[:,1]
    except Exception as e:
        pass

    print ("\n모델 보고서")
    try:
        print ("정확도(Accuracy) : %.4g" % metrics.accuracy_score(df[y_columns].values, df_predictions))
    except Exception as e:
        pass

    try:
        print ("AUC 점수 (Train): %f" % metrics.roc_auc_score(df[y_columns].values, df_predprob))
    except Exception as e:
        pass

    if performCV:
        try:
            cv_score = cross_validation.cross_val_score(model, df[X_columns], df[y_columns], cv=cv_folds, scoring='roc_auc')
            print ("교차검증(CV) 점수 : 평균 - %.7g | 표준편차 - %.7g | 최소값 - %.7g | 최대값 - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        except Exception as e:
            pass
        
    try:
        if printFeatureImportance:
            fig, ax = plt.subplots(1, 1, sharex=True)
            feat_imp = pd.Series(model.feature_importances_, X_columns).sort_values(ascending=False)
            feat_imp.plot(ax=ax, kind='bar', title='특성(Feature) 중요도')
            ax.yaxis.set_major_formatter(percent_formatter)
            plt.ylabel('특성(Feature) 중요도 점수')
            plt.savefig('특성(Feature) 중요도.png')
            plt.show()
    except Exception as e:
        pass

def XGBoost_TrainingPerformance(model, X, y, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=verbose)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.yaxis.set_major_formatter(percent_formatter)
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.yaxis.set_major_formatter(percent_formatter)
    ax.legend()
    plt.ylabel('Classification Error')
    plt.title('XGBoost Classification Error')
    plt.show()

def XGBoost_EarlyStop(model, X, y, early_stopping_rounds=10, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric="logloss", eval_set=eval_set, verbose=verbose)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def XGBoost_NumberOfThreads(X, y, num_threads=[1, 2, 3, 4,5,6,7,8]):
    results = []
    for n in num_threads:
        start = time.time()
        model = XGBClassifier(nthread=n)
        model.fit(X, y)
        elapsed = time.time() - start
        print(n, elapsed)
        results.append(elapsed)

    plt.plot(num_threads, results)
    plt.ylabel('Speed (seconds)')
    plt.xlabel('Number of Threads')
    plt.title('XGBoost Training Speed vs Number of Threads')
    plt.show()

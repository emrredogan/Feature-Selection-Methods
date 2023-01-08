import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

class FeatureSelection:
    def __init__(self, df, X_train, X_test, y_train, y_test):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    @staticmethod
    def df_rename(df):
        '''
        Fix feature names according to python format
        '''

        if df.columns is not None:
            try:
                df.drop("Patient Id", axis=1, inplace=True)
                old_column_names = df.columns
                column_mapping = {old_name: old_name.lower().replace(
                    " ", "_") for old_name in old_column_names}
                df = df.rename(columns=column_mapping)
                df["level"].replace(
                    {'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
            except:
                pass
        return df

    @staticmethod
    def descriptive_stats(df):
        df_describe = df.describe().T
        return df_describe


    @staticmethod
    def check_missing(df):
        '''
        Check missing values of dataframe
        '''

        df_missing = df.isnull().sum()
        return df_missing


    @staticmethod
    def df_info(df):
        '''
        Return info of dataframe
        '''
        return df.info()


    @staticmethod
    def df_corr_heatmap(df):
        '''
        Draw correlation matrix
        '''

        plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), annot=True, cmap=plt.cm.PuBu)
        plt.show()


    @staticmethod
    def df_multicollinearity_elimination(df):
        '''
        Remove highly correlated feature (above .80) that has lower correlation with target.
        '''

        corr = df.corr(method="pearson")
        drop_columns_list = []

        for i, column in enumerate(df.columns):
            highly_correlated_features = df.columns[(corr[column].abs() > 0.70) & (corr[column].index != column)]
            highly_correlated_features = list(highly_correlated_features)

            if len(drop_columns_list) > 0:
                drop_columns_list.extend(highly_correlated_features)
            else:
                drop_columns_list = highly_correlated_features

        drop_columns_list = list(drop_columns_list)

        for column in drop_columns_list:
            if column in df.columns:
                other_column = [col for col in drop_columns_list if col != column][0]
                try:
                    if abs(corr["level"][column]) > abs(corr["level"][other_column]):
                        df = df.drop(columns=other_column)
                    else:
                        df = df.drop(columns=column)
                except:
                    pass
        return df


    @staticmethod
    def train_test_split(df):
        '''
        Train Test Split Method
        '''

        X = df.drop('level', axis=1)
        y = df.level

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.20, random_state=123)
        return X_train, X_test, y_train, y_test


    def step_forward_feature_selection(self):
        '''
        Using step forword feature selection. Adjust bakward method by changing "forward" parameter.
        '''

        sfs = SequentialFeatureSelector(LogisticRegression(n_jobs=-1),
                                        scoring='accuracy', forward=True, verbose=2, cv=3, k_features=round(self.X_train.shape[1]/2), n_jobs=-1).fit(self.X_train, self.y_train)

        return sfs.k_feature_names_, sfs.k_feature_idx_, pd.DataFrame.from_dict(sfs.get_metric_dict()).T


    def exhaustive_selection(self, X_train):
        '''
        Using exhaustive selection method
        '''
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        efs = ExhaustiveFeatureSelector(LogisticRegression(solver='newton-cg', n_jobs=-1),
                                        scoring='accuracy', min_features=1, max_features=5 ,cv=6, n_jobs=-1).fit(X_train_scaled, self.y_train)

        return efs.best_feature_names_, efs.best_idx_, pd.DataFrame.from_dict(efs.get_metric_dict()).T


    def logistic_reg_with_best_feat(self, X_train):
        '''
        Build Logistic regression with best features according to feature selection methods
        '''

        logreg = LogisticRegression(
            penalty='l2', fit_intercept=True, verbose=0, n_jobs=-1, multi_class='ovr').fit(X_train, self.y_train)

        return logreg


    def performance_logistic(self, clf, X_test):
        '''
        Return gini value
        '''

        lb = LabelBinarizer()
        y_test_onehot = lb.fit_transform(self.y_test)
        y_pred_proba = clf.predict_proba(X_test)
        gini_coefficients = []
        for i in range(len(self.y_test)):
            y_true = y_test_onehot[i]
            y_pred = y_pred_proba[i]
            auc = metrics.roc_auc_score(y_true, y_pred, average=None)
            gini = 2 * np.mean(auc) - 1
            gini_coefficients.append(gini)
        mean_gini = np.mean(gini_coefficients)
        
        return mean_gini

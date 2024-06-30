
# Part 1: Decision Trees with Categorical Attributes
# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
#%%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#%%
def read_csv_1(data_file):
    df = pd.read_csv(data_file)
    df=df.drop('fnlwgt',axis=1)
    return df
#%%
# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	num_instances = df.shape[0]
	return num_instances
#%%
# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	attribute_names=df.columns.to_list()
	return attribute_names
#%%
# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	num_missing_values = df.isnull().sum().sum()
	return num_missing_values
#%%
# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	columns_with_missing_values = df.columns[df.isnull().any()].tolist()
	return columns_with_missing_values
#%%
# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
    total_instances = len(df)
    bachelors_or_masters_instances = len(df[(df['education'] == 'Bachelors') | (df['education'] == 'Masters')])
    percentage = (bachelors_or_masters_instances / total_instances) * 100
    percentage = "{:.1f}%".format(percentage)
    return percentage
#%%
# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df=df.dropna()
	return df
#%%
# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(df[categorical_columns])
    onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
    X_encoded = pd.DataFrame(onehot_encoded, columns=onehot_feature_names)
    return X_encoded
#%%
# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    y_encoded = df['class']
    return y_encoded
#%%
# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    return y_pred_train
#%%
# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("Lengths of y_pred and y_true must be the same")
    error_rate = 1 - accuracy_score(y_true, y_pred)
    return error_rate



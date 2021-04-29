from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import pandas as pd


labeled_df = pd.read_csv('labeled_data.csv', encoding='utf-8')

from utils import str_columns_to_one_hot

labeled_df = str_columns_to_one_hot(labeled_df.drop('Application ID', axis=1))

x = labeled_df.drop(['r', 'p_1', 'p_2', 'p_3'], axis=1)
y = labeled_df['r']

mean_x = x.mean()
std_x = x.std()

x = ((x - mean_x) / std_x)

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.25)
clf = XGBClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_validation)
print(f1_score(y_validation, y_pred))



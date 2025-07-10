from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

employee_df = pd.read_csv('C:/Users/vyshn/Documents/MTech Integrated/3rd_year/Sem-6/SPM/Project/Employee Attrition.csv')

#separate the categorical and numerical column
X_categorical = employee_df.select_dtypes(include=['category'])
X_numerical = employee_df.select_dtypes(include=['int64'])
y = employee_df['Attrition']


#handle the categorical variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

X_categorical = onehotencoder.fit_transform(X_categorical).toarray()
X_categorical = pd.DataFrame(X_categorical)
X_categorical
     

#concat the categorical and numerical values

X_all = pd.concat([X_categorical, X_numerical], axis=1)
X_all.head()
     
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Convert column names to strings
X_all.columns = X_all.columns.astype(str)

# Now scale the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

# Assuming you have already trained your logistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(logistic_regression_model, 'logistic_regression_model.pkl')

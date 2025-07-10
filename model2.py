import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

def train_model(df):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Create a pipeline with preprocessing and logistic regression
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    
    # Save the model
    joblib.dump(pipeline, 'attrition_model.pkl')
    
    return accuracy

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('C:/Users/vyshn/Documents/MTech Integrated/3rd_year/Sem-6/SPM/Project/Employee Attrition.csv')
    
    # Train the model
    accuracy = train_model(df)

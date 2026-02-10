import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  StandardScaler
def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()

    df['Car_Age'] = 2024 - df['Year']
    df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
    numeric_cols = X.select_dtypes(exclude='object').columns

    num_pipeline = Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())

    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_cols),
            ('num', num_pipeline, numeric_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor

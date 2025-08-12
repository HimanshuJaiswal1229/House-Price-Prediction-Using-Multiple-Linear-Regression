

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/kaggle/data_description.csv")

X = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = train_df['SalePrice']

train_df = pd.read_csv("/kaggle/train.csv")
test_df = pd.read_csv("/kaggle/test.csv")

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_df[features]
y = train_df['SalePrice']

X = X.dropna()
y = y.loc[X.index]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R² Score:", r2_score(y_val, y_pred))

X_test = test_df[features].fillna(0)
test_preds = model.predict(X_test)

output = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_preds})
output.to_csv("/kaggle/sample_submission.csv", index=False)
print("Predictions saved to /kaggle/sample_submission.csv")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost


df = pd.read_csv('/home/mwanikii/Desktop/car-sales.csv')

#Using simpleimputer to fill NaN values in Odometer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
#Fitting the data to the imputer object
imputer = imputer.fit(df[['Odometer (KM)']])
df['Odometer (KM)'] = imputer.transform(df[['Odometer (KM)']])

#Removing null values in the column
null_val = df['Make'].isnull()
#print(null_val) = 49
df1 = df.dropna()
df = df1

#Using encoders
#Categorical_values =  'Make','Colour'
LE =  LabelEncoder()
df['Make_Encoded'] = LE.fit_transform(df['Make'])
df['Colour_Encoded'] = LE.fit_transform(df['Colour'])

print(df['Make_Encoded'].describe())
print(df['Colour_Encoded'].describe())
print(df['Doors'].describe())


#Defining X and y
y = df['Price']

#Scaling values in Odometer(KM)
scaler = MinMaxScaler()
X_drop = df.drop(['Price','Make','Colour','Odometer (KM)'], axis=1)
X_array = np.array(X_drop) 
X_scaled = scaler.fit_transform(df[['Odometer (KM)']])
#X = np.append(X_array, X_scaled, axis=1)

#Trying different approach to get more accurate values
scaler = MinMaxScaler()
X_drop_1 = df.drop (['Price', 'Make', 'Colour'], axis=1)
X = scaler.fit_transform(X_drop_1)


#Graph1
plt.bar(df['Make'], df['Price'])
plt.title('Make and Price comparison')
plt.xlabel('Make')
plt.ylabel('Price')
#plt.show()

#Graph2 
plt.scatter(df['Odometer (KM)'], df['Price'])
plt.title('KM and Price comparison')
plt.xlabel('KM')
plt.ylabel('Price')
#plt.show()

#Graph3 
plt.scatter(df['Colour_Encoded'], df['Price'])
plt.title('Colour and Make Comparison')
plt.xlabel('Colour')
#plt.show()

#Graph4
plt.bar(df['Colour'], df['Price'])
plt.title('Colour and Make Comparison')
plt.xlabel('Colour')
plt.ylabel('Price')
#plt.show()

print (X)

#Making train_test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=12)

#Converting y_test into one dimensional values for accuracy test
y_1D = y_test.values

#Naive classifier for comparison with MAE and accuracy_score
dm = DummyClassifier(strategy='most_frequent')
dm.fit(X_train, y_train) 
y_compare = dm.predict(X_test)

#Multiple models
def train_ml_model(X, y, model):
    if model == 'lr':
        
        model = LinearRegression()

    elif model == 'rf':

        model = RandomForestRegressor()

    elif model == 'lsvc':

        model = LinearSVC()
    
    elif model == 'xg':

        model = xgboost

    model.fit(X_train, y_train)

    return model

#inherits models and predicts
def predict(training, model):
    training = train_ml_model(X, y, model)
    predict  = training.predict(X_test)

    return print(predict)


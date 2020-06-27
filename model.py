import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("C:/Users/Navee/OneDrive/Desktop/insurance.csv")


def categorical_encoding(data, columns):
    for cols in columns:
        data = pd.concat([data, pd.get_dummies(data[cols]).rename(columns=lambda x: cols + '_' + str(x))], axis=1)

        data = data.drop(cols, axis=1)

    return data


categorical_columns = list(data.columns[data.dtypes == 'object'])

new_data = categorical_encoding(data, categorical_columns)

X_train_org, X_test_org, y_train, y_test = train_test_split(new_data.drop('charges', axis=1), new_data['charges'],
                                                            random_state=0)

sc = StandardScaler()

#X_train = sc.fit_transform(X_train_org)
#X_test = sc.transform(X_test_org)

lr_poly = LinearRegression()

#poly = PolynomialFeatures(degree=2)
#X_train_poly = poly.fit_transform(X_train_org)
#X_test_poly = poly.transform(X_test_org)
lr_poly.fit(X_train_org, y_train)

#listo=np.array([20, 0, 1, 10, 1, 0, 1, 0, 0, 0, 1]).reshape(1,-1)
#print(listo.shape)
#print(lr_poly.predict(listo))

pickle.dump(lr_poly, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

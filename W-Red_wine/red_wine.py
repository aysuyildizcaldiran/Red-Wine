# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score , mean_absolute_error,mean_squared_error 
from sklearn.metrics import median_absolute_error, r2_score
from scipy import stats

#data read
data=pd.read_csv('winequality-red.csv')
data.head()
data.quality.replace([3, 4, 5, 6, 7, 8], [0, 0, 0, 1, 1, 1], inplace=True)
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

'''
#kayip,unique veriler
import re 
kayip_veriler=[]
sayisal_olmayan_veriler=[]

for oznitelik in data:
    essizdeger=data[oznitelik].unique()
    print("'{}' ozniteliğine sahip unique deger {}".format(oznitelik,essizdeger.size))
    
    if(True in pd.isnull(essizdeger)):
        s="{} ozniteliğe ait kayip veriler {}".format(oznitelik,pd.isnull(data[oznitelik]).sum())
        kayip_veriler.append(s)
    for i in range(1,np.prod(essizdeger.shape)):
       if(re.match('nan',str(essizdeger[i]))): 
           break
       if not(re.search('(^/d+\d*$)|(^\d*\.?\d+$)',str(essizdeger[i]))):
           sayisal_olmayan_veriler.append(oznitelik)
           break 
       
print("Kayıp veriye sahip oznitelikler:\n{}\n\n".format(kayip_veriler))
print("Sayısal olmayan veriye sahip oznitelikler:\n{}".format(sayisal_olmayan_veriler))

'''

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=0)
'''
#LineerRegression
lr=LinearRegression()
lr.fit(X_train, Y_train) 
tahmin=lr.predict(X_test)
plt.scatter(X_train, Y_train,color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.show()
plt.scatter(X_test, Y_test,color='red')
plt.plot(X_train,lr.predict(X_train),color='blue')
plt.show()
print('Eğim(Q1):',lr.coef_)
print('Kesen(Q0):',lr.intercept_)
print("y=%0.2f"%lr.coef_+"x+%0.2f"%lr.intercept_)
print("R-Kare:",r2_score(Y_test,tahmin))
print("MAE:",mean_absolute_error(Y_test,tahmin))
print("MSE:",mean_squared_error( Y_test,tahmin))
print("MedAE:",median_absolute_error(Y_test,tahmin))
'''
'''
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)
lineer2=LinearRegression()
lineer2.fit(x_poly,y)
plt.scatter(x,y,color='red')
plt.plot(x,lineer2.predict(poly.fit_transform(x)),color='blue')
plt.show()

'''
'''
#Bağımsız
print(stats.describe(X_train,axis=0))
print(np.std(X_train,axis=0))
print(stats.describe(X_test,axis=0))
print(np.std(X_test,axis=0))
#Bağımlı
print(stats.describe(Y_train,axis=0))
print(np.std(Y_train,axis=0))
print(stats.describe(Y_test,axis=0))
print(np.std(Y_test,axis=0))
'''

#Öznitelik Ölçeklendirme
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
Y_train=scaler.fit_transform(Y_train.reshape(-1,1))
Y_test=scaler.fit_transform(Y_test.reshape(-1,1))

'''
#Eğitim setinin Çoklu regresyona uyarlanması
model_regresyon=LinearRegression()
model_regresyon.fit(X_train,Y_train)
y_pred=model_regresyon.predict(X_test)

X=np.append(arr=np.ones((1599,1)).astype(int),values=x,axis=1) #1 değerli kolon eklendi

import statsmodels.api as sm 
X_yeni=X[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
model_regresyon_OLS=sm.OLS(endog=y,exog=X_yeni).fit()
print(model_regresyon_OLS.summary())

X_iki=X[:,[1,2,3,4,5,6,7,8,9,10,11]]
model_regresyon_OLS=sm.OLS(endog=y,exog=X_iki).fit()
print(model_regresyon_OLS.summary())

X_uc=X[:,[2,3,4,5,6,7,8,9,10,11]]
model_regresyon_OLS=sm.OLS(endog=y,exog=X_uc).fit()
print(model_regresyon_OLS.summary())

X_dort=X[:,[2,4,5,6,7,8,9,10,11]]
model_regresyon_OLS=sm.OLS(endog=y,exog=X_dort).fit()
print(model_regresyon_OLS.summary())

X_bes=X[:,[2,5,6,7,8,9,10,11]]
model_regresyon_OLS=sm.OLS(endog=y,exog=X_bes).fit()
print(model_regresyon_OLS.summary())


print("R-Kare=%0.2f"%r2_score(Y_test,y_pred))
print("MAE=%0.2f"%mean_absolute_error(Y_test,y_pred))
print("MSE=%0.2f"%mean_squared_error(Y_test,y_pred))
print("MedAE=%0.2f"%median_absolute_error(Y_test,y_pred))
print("RMSE=%0.2f"%np.sqrt(mean_squared_error(Y_test,y_pred)))
'''


'''
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

'''


'''
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)

from sklearn.model_selection import cross_val_score
val_score_sgd = cross_val_score(sgd_clf, X_train_scaled,Y_train,scoring='accuracy',cv=10)
val_score_sgd.mean()
'''



from sklearn.ensemble import RandomForestClassifier
forest_clf=RandomForestClassifier(random_state=42)
forest_clf.fit(X_train,Y_train)
y3_pred=forest_clf.predict(X_test)
#y3_pred=scaler.inverse_transform(y3_pred.reshape(-1,1))


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X_train,Y_train)
y2_pred=rf_reg.predict(X_test)
#y2_pred=scaler.inverse_transform(y2_pred.reshape(-1,1))


from sklearn.metrics import roc_curve,auc
#hm=confusion_matrix(Y_test, y2_pred)
#rapor=classification_report(Y_test, y2_pred)

ypo,dpo,esikDeger=roc_curve(Y_test,y2_pred)
aucDegeri=auc(ypo,dpo)
plt.figure()
plt.plot(ypo,dpo,label='AUC %0.2f'%aucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.title('Regression')
plt.legend(loc="best")
plt.show()



#hm2=confusion_matrix(Y_test, y3_pred)
#rapor=classification_report(Y_test, y3_pred)

ypo,dpo,esikDeger=roc_curve(Y_test,y3_pred)
aucDegeri=auc(ypo,dpo)
plt.figure()
plt.plot(ypo,dpo,label='AUC %0.2f'%aucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.title('Classifier')
plt.legend(loc="best")

plt.show()

print("R-Kare=%0.2f"%r2_score(Y_test,y2_pred))
print("MAE=%0.2f"%mean_absolute_error(Y_test,y2_pred))
print("MSE=%0.2f"%mean_squared_error(Y_test,y2_pred))
print("MedAE=%0.2f"%median_absolute_error(Y_test,y2_pred))
print("RMSE=%0.2f"%np.sqrt(mean_squared_error(Y_test,y2_pred)))

print("R-Kare=%0.2f"%r2_score(Y_test,y3_pred))
print("MAE=%0.2f"%mean_absolute_error(Y_test,y3_pred))
print("MSE=%0.2f"%mean_squared_error(Y_test,y3_pred))
print("MedAE=%0.2f"%median_absolute_error(Y_test,y3_pred))
print("RMSE=%0.2f"%np.sqrt(mean_squared_error(Y_test,y3_pred)))
# Heart Disease Prediction Using Logistic Regression

import pandas as pd

import pylab as pl

import numpy as np

import scipy.optimize as opt

import statsmodels.api as sm

from sklearn import preprocessing

'exec(% matplotlib inline)'

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns






disease_df = pd.read_csv("Heart_Disease_Prediction.csv")

disease_df.rename(columns ={'male':'Sex_male'}, inplace = True)



disease_df.dropna(axis = 0, inplace = True)

print(disease_df.head(), disease_df.shape)

![Screenshot 2024-04-08 204601](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/960cb837-79fd-4806-82e1-a59fd6636925)


disease_df.isnull().sum()

![Screenshot 2024-04-08 204633](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/323eedc5-bd3f-49b6-9d02-cbadbcde3de9)


count=0

for i in disease_df.isnull().sum(axis=1):

    if i>0:
        count=count+1
        
print('Total number of rows with missing values is ', count)

print('since it is only',round((count/len(disease_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')

![Screenshot 2024-04-08 204707](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/1e30b3a4-9e21-420e-a73c-8d046f762015)

disease_df.dropna(axis=0,inplace=True)

def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))
    
    for i, feature in enumerate(features): 
    
        ax=fig.add_subplot(rows,cols,i+1)
        
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    
    plt.show()
    
draw_histograms(disease_df,disease_df.columns,6,3)


![Screenshot 2024-04-08 204843](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/ff18ed65-1aa5-4399-920c-d7f6e823e03f)

disease_df.describe()

![Screenshot 2024-04-08 204919](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/7a28d853-3961-4102-b5c3-6fda35f004f6)

X = np.asarray(disease_df[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120']])

y = np.asarray(disease_df['Heart Disease'])

#normalization of the dataset

X = preprocessing.StandardScaler().fit(X).transform(X)

#Train-and-Test -Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( 

        X, y, test_size = 0.3, random_state = 4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

![Screenshot 2024-04-08 204944](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/0a746dbb-d267-4cd7-afa9-5c6c1a37c795)


plt.figure(figsize=(7, 5))

sns.countplot(x='Heart Disease', data=disease_df,
             palette="BuGn_r")
             
plt.show()


![Screenshot 2024-04-08 205014](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/d69939dd-4d3b-4b12-81d6-7aa893f89c71)

laste = disease_df['Thallium'].plot()

plt.show(laste)

![Screenshot 2024-04-08 205103](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/b2b9ea09-6c68-4ad2-9327-578c025ebfbe)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score

print('Accuracy of the model is =', 

      accuracy_score(y_test, y_pred))

      ![Screenshot 2024-04-08 205130](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/55a0c83c-34b2-4ae7-ac18-3dd143f9c9d2)

      

      from sklearn.metrics import confusion_matrix, classification_report
      
cm = confusion_matrix(y_test, y_pred)

conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['Predicted:0', 'Predicted:1'], 
                           index =['Actual:0', 'Actual:1'])

plt.figure(figsize = (8, 5))

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")


plt.show()

print('The details for confusion matrix is =')

print (classification_report(y_test, y_pred))

![Screenshot 2024-04-08 205243](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/82a0f0b6-0c41-4566-9021-6e35a70497d2)


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)

![Screenshot 2024-04-08 205316](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/dad4be61-1754-4ac0-92c0-70bea3c2fd0a)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Assuming you have predictions (y_pred) from your model

y_pred = logreg.predict(X_test)  # Replace with your actual predictions

#Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, pos_label='Presence')

recall = recall_score(y_test, y_pred, pos_label='Presence')

f1 = f1_score(y_test, y_pred, pos_label='Presence')

conf_matrix = confusion_matrix(y_test, y_pred, labels=['Absence', 'Presence'])

print(f"Accuracy: {accuracy:.2f}")

print(f"Precision: {precision:.2f}")

print(f"Recall: {recall:.2f}")

print(f"F1-score: {f1:.2f}")

print("Confusion Matrix:")

print(conf_matrix)


![Screenshot 2024-04-08 205341](https://github.com/BabuBharathB/machine-learning-project/assets/167573509/d87ec009-1ef7-48bc-9175-cd83a9453fce)






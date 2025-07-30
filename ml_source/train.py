import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor #randomforest comes under ensemble
from sklearn.model_selection import train_test_split #to train the model
import joblib 


df=pd.read_csv("D:\\proj2\\Data\\pneumonia_covid_diagnosis_dataset.csv")
#perform EDA

columns=["Gender","Fever","Cough","Fatigue","Breathlessness","Comorbidity","Type","Stage"]
for col in columns:
    le=LabelEncoder()#LabelEncoder converts text into numbers
    df[col]=le.fit_transform(df[col])

print(df.head())
df=df.drop("Is_Curable",axis=1)

print(df.head())
#push the data in our model

x=df.drop(columns=["Survival_Rate"],axis=1)
y=df['Survival_Rate']
#train and test the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#prediction
model=RandomForestRegressor()
model.fit(x_train,y_train)

prd=model.predict(x_test)
#save our model using pkl file because machine learnig
joblib.dump(model,"covid_diag.pkl")
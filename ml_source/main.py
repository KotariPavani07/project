from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

#load the pkl file
data=joblib.load("covid_diag.pkl")

#create a class
class inp_data(BaseModel):
    Age:int
    Gender:int
    Fever:int
    Cough:int
    Fatigue:int
    Breathlessness:int
    Comorbidity:int
    Stage:int
    Type:int
    Tumor_Size:float

app=FastAPI()

@app.get("/")
def root_msg():
    return {"Message":"Welcome to hyd"}
@app.post("/predict")
def prediction(Data:inp_data):
    #convert everything into data fraame
    #inp=pd.DataFrame([Data.dict()])#data into dict values like key value 
    inp=np.array([[Data.Age,Data.Gender,Data.Fever,Data.Cough,Data.Fatigue,
    Data.Breathlessness,Data.Comorbidity,Data.Stage,Data.Type,Data.Tumor_Size]])
    prdd=data.predict(inp)[0]
    return {"Prediction":prdd}



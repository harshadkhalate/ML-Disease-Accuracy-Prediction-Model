from typing import List
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('./kal.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

disease_symptoms = {
    'Allergy': ['eye_irritation','running_nose','stuffy_nose','watery_eyes','sneezing','itchy_nose','itchy_throat','inflamed_throat'],
    'Diarrhea': ['watery_stools','frequent_bowel_movements','abdomen_pain','nausea','bloating','bloody_stools','fever'],
    'Cold_and_flue': ['headachae','more_intense_pain','fatigue','dry_cough','sore_throat','cough'],
    'Stomachae': ['vomiting','heartburn','indigestion','change_in_apetite','anemia'],
    'Dengue': ['rashes','pain_behind_eyes','pain_in_joints','fever','nausea','headachae'],
    'Malaria': ['pain_in_joints','feeling_of_discomfort','watery_stools','frequent_bowel_movements','abdomen_pain','nausea','vomiting','fever'],
    'Pneumonia': ['low energy','cough_with_mucus','greenish_yellow_bloody_mucus','shortness_of_breath','chills','sweating','shallow_breathing','chest_pain','fever','fatigue','change_in_apetite']
}

class Symptomclass(BaseModel):
    symptoms: List[str]

@app.post("/predict")
def func(s: Symptomclass):
    input_symptoms = s.symptoms
    input_symptoms_set = set(input_symptoms)
    max_accuracy = 0
    predicted_disease = None

    for disease, symptoms in disease_symptoms.items():
        disease_symptoms_set = set(symptoms)
        accuracy = len(input_symptoms_set.intersection(disease_symptoms_set)) / len(disease_symptoms_set) * 100
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            predicted_disease = disease

    return {"prediction": predicted_disease, "Accuracy": max_accuracy}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3004)

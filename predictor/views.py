from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
import os
from rest_framework.response import Response
from rest_framework.decorators import api_view

# Load models and encoder safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from django.http import JsonResponse

def predict_prognosis(request):
    if request.method == 'GET':
        symptoms = request.GET.getlist('symptoms')
        # Process the symptoms and return prediction
        return JsonResponse({"prediction": "some_disease"})
    return JsonResponse({"error": "Method not allowed"}, status=405)


def load_and_fix_model(model_path):
    """Loads and re-saves the model to prevent version issues."""
    model = joblib.load(model_path)
    joblib.dump(model, model_path)  # Re-save to ensure compatibility
    return model

model_prognosis = load_and_fix_model(os.path.join(BASE_DIR, 'model_prognosis.pkl'))
model_time = load_and_fix_model(os.path.join(BASE_DIR, 'model_time.pkl'))
label_encoder = load_and_fix_model(os.path.join(BASE_DIR, 'label_encoder.pkl'))

# Load symptom columns from training dataset
train_data = pd.read_csv(os.path.join(BASE_DIR, 'Training_with_estimated_time.csv'))
symptoms = train_data.drop(columns=['prognosis', 'estimated_time']).columns.tolist()  # Convert to list for safety

@api_view(['GET'])
def predict_prognosis(request):
    try:
        symptom_inputs = request.GET.getlist('symptoms')  # Read symptoms from query params

        # Ensure valid symptoms are selected
        selected_symptoms = [symptom for symptom in symptom_inputs if symptom in symptoms]

        # Create DataFrame with all symptoms set to 0
        input_data = pd.DataFrame(np.zeros((1, len(symptoms))), columns=symptoms)

        # Set provided symptoms to 1
        input_data.loc[0, selected_symptoms] = 1

        # Make predictions
        prognosis_encoded = model_prognosis.predict(input_data)[0]
        prognosis = label_encoder.inverse_transform([prognosis_encoded])[0]
        estimated_time = model_time.predict(input_data)[0]

        result = {
            "prognosis": prognosis,
            "estimated_time": round(float(estimated_time), 2)
        }
        return Response(result)

    except Exception as e:
        return Response({"error": str(e)}, status=400)

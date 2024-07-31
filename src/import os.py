import os
from vertexai.generative_models import (
    GenerativeModel,
    FunctionDeclaration,
    Tool,
    Part,
    Content
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "KEY.json"

patient_data = {
    "id": 1,
    "name": "Alice Smith",
    "address": "123 Maple Street, Springfield, IL",
    "age": 34,
    "dob": "1989-05-14",
    "gender": "Female",
    "blood_type": "O+",
    "height": 165,
    "weight": 60,
    "mother_name": "Emily Smith",
    "mother_dob": "1965-03-30",
    "mother_death_date": None,
    "father_dob": "1963-07-25",
    "father_death_date": None,
    "mother_diseases": "Hypertension",
    "father_diseases": "Diabetes",
    "health_checkup_date": "2024-07-15",
    "temperature": 98.6,
    "blood_pressure": "120/80",
    "heart_rate": 72,
    "cholesterol": 190,
    "blood_sugar": 85,
    "bmi": 22.1,
    "allergies": "Pollen",
    "body_oxygen": 98,
    "respiratory_rate": 16,
    "physical_injuries": "None"
}

model = GenerativeModel("gemini-1.5-pro-001")

# Define the function for detecting abnormalities in blood pressure and heart rate
detect_abnormal_health_metrics_func = FunctionDeclaration(
    name="detect_abnormal_health_metrics",
    description="Detects abnormalities in blood pressure and heart rate based on standard thresholds.",
    parameters={
        "type": "object",
        "properties": {
            "blood_pressure": {
                "type": "string",
                "description": "Patient's blood pressure reading in the format 'systolic/diastolic'."
            },
            "heart_rate": {
                "type": "integer",
                "description": "Patient's heart rate in beats per minute."
            },
            "past_history": {
                "type": "object",
                "properties": {
                    "mother_diseases": {
                        "type": "string",
                        "description": "Mother's medical history."
                    },
                    "father_diseases": {
                        "type": "string",
                        "description": "Father's medical history."
                    }
                },
                "required": ["mother_diseases", "father_diseases"]
            }
        },
        "required": ["blood_pressure", "heart_rate", "past_history"]
    }
)

# Create the tool
health_metrics_tool = Tool(
    function_declarations=[detect_abnormal_health_metrics_func]
)

# Generate content using the tool
user_prompt = {
    "blood_pressure": patient_data["blood_pressure"],
    "heart_rate": patient_data["heart_rate"],
    "past_history": {
        "mother_diseases": patient_data["mother_diseases"],
        "father_diseases": patient_data["father_diseases"]
    }
}

response = model.generate_content(
    str(user_prompt),
    tools=[health_metrics_tool],
)

print(response)

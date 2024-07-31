import os
from vertexai.generative_models import (
    GenerativeModel,
    FunctionDeclaration,
    Tool,
    Part, 
    Content
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/riddhigohil/config/riddhi-gohil-intern-cd51d97234ea.json"

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


detect_abnormal_blood_pressure_func = FunctionDeclaration(
    name="detect_abnormal_blood_pressure",
    description="Detects abnormal blood pressure readings based on standard thresholds.",
    parameters={
        "type": "object",
        "properties": {
            "blood_pressure": {
                "type": "string",
                "description": "Tells whether the blood pressure is normal, high or low. Blood pressure is considered high if systolic > 140 or diastolic > 90. It's considered low if systolic < 90 or diastolic < 60. Otherwise it is considered to be normal.",
                "enum":["low","high","normal"]
            },
            "dieases_predicted":{
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Predicted diseases (if any) based on the given blood pressure, past self history and family history."
                }
            }
        },
        "required": ["blood_pressure","dieases_predicted"]
    }
)

abnormalities_tool = Tool(
    function_declarations=[detect_abnormal_blood_pressure_func]
)

user_prompt = "Is there any abnormality in my health record?"

response = model.generate_content(
    user_prompt,
    tools=[abnormalities_tool],
)
print(response)

# params = {}
# for key, value in response.candidates[0].content.parts[0].function_call.args.items():
#     params[key[9:]] = value
    

# response = model.generate_content(
#     [
#     Content(role="user", parts=[
#         Part.from_text(user_prompt + """Give your answer in steps with lots of detail
#             and context, including any abnormalities in blood pressure."""),
#     ]),
#     Content(role="function", parts=[
#         Part.from_dict({
#             "function_call": {
#                 "name": "detect_abnormal_blood_pressure_func",
#             }
#         })
#     ]),
#     Content(role="function", parts=[
#         Part.from_function_response(
#             name="get_exchange_rate",
#             response={
#                 "content": patient_data.text,
#             }
#         )
#     ]),
#     ],
#     tools=[abnormalities_tool],
# )


# print(response.candidates[0].content.parts[0].text)
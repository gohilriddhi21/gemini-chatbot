import os
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/riddhigohil/config/riddhi-gohil-intern-cd51d97234ea.json"

PROJECT_ID = "riddhi-gohil-intern"
LOCATION = "us-central1"
MODEL_ID = "gemini-1.5-pro"

user_message = "We need someone to help set up a Kubernetes cluster. Any ideas?"
content = "Eric Hole is skilled in Kubernetes, GKE, Cloud Native. Will Beebe is skilled in Kubernetes, Cloud Run, and Terraform. Sean Gilley is skilled in generative AI and being a baller."
   
model = GenerativeModel(model_name=MODEL_ID)
response = model.generate_content(
    [
        content,user_message
    ]
)
print(response.text)


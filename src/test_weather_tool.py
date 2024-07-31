import os
from vertexai.generative_models import (
    GenerativeModel,
    FunctionDeclaration,
    Tool,
    Content,
    Part
)
import requests

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/riddhigohil/config/riddhi-gohil-intern-cd51d97234ea.json"

model = GenerativeModel("gemini-1.5-pro-001")

exchange_rate_func = FunctionDeclaration(
    name="get_exchange_rate",
    description="Get the exchange rate for currencies between countries",
    parameters={
    "type": "object",
    "properties": {
        "currency_date": {
            "type": "string",
            "description": "A date that must always be in YYYY-MM-DD format or the value 'latest' if a time period is not specified"
        },
        "currency_from": {
            "type": "string",
            "description": "The currency to convert from in ISO 4217 format"
        },
        "currency_to": {
            "type": "string",
            "description": "The currency to convert to in ISO 4217 format"
        }
    },
         "required": [
            "currency_from",
            "currency_date",
      ]
  },
)

exchange_rate_tool = Tool(
    function_declarations=[exchange_rate_func]
)

user_prompt = "What's the exchange rate from rupees to US dollars today? How much is 200 RS in USD?"

# response = model.generate_content(
#     user_prompt
# )
# print(response.text)

# response = model.generate_content("""
#         Your task is to extract parameters from the user's input and return it as a
#         structured JSON payload. The user will ask about the exchange rate and which
#         currency they are converting from and converting to.

#         User input: {user_prompt}

#         Please extract the currencies as parameters and put them in a JSON object.
#         Attributes of JSON object are from_currency and to_currency
        
#         """.format(user_prompt=user_prompt)) 
# print(response.text)


response = model.generate_content(
    user_prompt,
    tools=[exchange_rate_tool],
)

# print([(k[9:], v) for k, v in response.candidates[0].content.parts[0].function_call.args.items()])

params = {}
for key, value in response.candidates[0].content.parts[0].function_call.args.items():
    params[key[9:]] = value
# print(params)


url = f"https://api.frankfurter.app/{params['date']}"
api_response = requests.get(url, params=params)
print(api_response.text)

response = model.generate_content(
    [
    Content(role="user", parts=[
        Part.from_text(user_prompt + """Give your answer in steps with lots of detail
            and context, including the exchange rate and date."""),
    ]),
    Content(role="function", parts=[
        Part.from_dict({
            "function_call": {
                "name": "get_exchange_rate",
            }
        })
    ]),
    Content(role="function", parts=[
        Part.from_function_response(
            name="get_exchange_rate",
            response={
                "content": api_response.text,
            }
        )
    ]),
    ],
    tools=[exchange_rate_tool],
)


print(response.candidates[0].content.parts[0].text)
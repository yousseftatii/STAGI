import requests
import os
api_key = os.getenv("HUGGINGFACE_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}"
}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)

print(response.json())


Deepseek api =  "sk-5229cbdfcb984c52b18447b202c464cf"
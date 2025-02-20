import requests

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-5229cbdfcb984c52b18447b202c464cf"

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "temperature": 0.7,
    "max_tokens": 50
}

response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
print(response.status_code)
print(response.json())
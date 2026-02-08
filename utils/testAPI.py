import requests

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

import requests
import json

url = "http://127.0.0.1:8003/api/v1/generate/preview"
payload = {
    "n_samples": 5,
    "modality": "tabular",
    "generator_type": "vae",
    "seed": 42
}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Successfully generated synthetic data preview!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Failed to generate preview. Status: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error: {e}")

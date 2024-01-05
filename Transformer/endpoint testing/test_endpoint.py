import requests
import os
import time

url = 'https://asl-demo.eastus2.inference.ml.azure.com/score'

tic = time.perf_counter()

files=[]
for i in range(128):
    files.append((f'image{i}',(f"{i}.jpg",open(os.path.join(f'test_endpoint_data',f'{i}.jpg'),'rb'),'application/octet-stream')))

# print(len(files))
headers = {
  'Authorization': ('Bearer '+'H0xsFsBzpEUDBzlaRZpI2of0btms1EAI'),
}
response = requests.request("POST", url, headers=headers, files=files)

print(response.request)
print(response.text)

toc = time.perf_counter()
print(f"Inference took {toc - tic:0.4f} seconds")
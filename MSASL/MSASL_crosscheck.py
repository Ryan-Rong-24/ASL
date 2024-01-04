from websockets import imports


import json

s1 = "MSASL_val.json"
s2 = "MSASL_train.json"

data1 = json.load(open(s1,'r'))
data2 = json.load(open(s2,'r'))

urls1 = []
for entry in data1:
    if entry['url'] not in urls1:
        urls1.append(entry['url'])

for entry in data2:
    if entry['url'] in urls1:
        print(entry['url'])

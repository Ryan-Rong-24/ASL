import pandas as pd
import json
df = pd.read_excel('dai-asllvd.xlsx')

data = []

columns = df.columns
labels = df['Main New Gloss.1']
links = df['Separate.1']

print(len(labels))
print(len(links))

for label,i in enumerate(labels):
    if "==" in label:
        continue
    
    # first create folder
    # download vids
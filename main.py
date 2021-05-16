import pandas as pd
from datetime import datetime
from mogafsc import nsga2


# Number of Clusters
K = 4

# Read DATA
df = pd.read_csv("dataset\house-vote-84.data", header=None)
df = df.drop([0], axis=1)

for column in df.columns:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes

data = df.to_numpy()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

nsga2(data, K)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

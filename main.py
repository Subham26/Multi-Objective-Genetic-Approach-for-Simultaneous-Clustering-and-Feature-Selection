from mogafsc import nsga2


# Number of Clusters
K = 4

# Read DATA
df = pd.read_csv("dataset\soybean-small.data", header=None)
data = df.to_numpy()

nsga2(data, K)

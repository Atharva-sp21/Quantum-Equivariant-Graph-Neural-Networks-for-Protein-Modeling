from dataset import ProteinDataset

dataset = ProteinDataset(["1UBQ.pdb"])

graph = dataset[0]
print(graph)
print("Label:", graph.y)

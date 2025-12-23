from parse_pdb import parse_pdb
from protein_graph import build_protein_graph

residue_types, coords = parse_pdb("1UBQ.pdb")
graph = build_protein_graph(residue_types, coords)

print(graph)
print("Nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)
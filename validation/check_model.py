import torch
import sys
import os

# Fix import path to look at the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hybrid_equivariant_gnn import HybridEquivariantGNN
from torch_geometric.data import Data

def get_random_rotation_matrix():
    """Generates a random 3D rotation matrix."""
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    return q

def test_equivariance():
    print("🧪 Starting Equivariance Test...")
    
    # 1. Define Edges (Simple chain: 0-1-2-3-4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4], 
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    num_edges = edge_index.shape[1] # Should be 8

    # 2. Create Dummy Data
    x = torch.randint(0, 5, (5, 1)).float()       # Node Features (Atomic Num)
    pos = torch.randn(5, 3)                       # 3D Coordinates
    edge_attr = torch.randn(num_edges, 1)         # <--- ADDED THIS (Edge Attributes)
    batch = torch.zeros(5, dtype=torch.long)      # Batch vector
    
    # Create the Data object with edge_attr
    data_original = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # 3. Initialize Model
    # Make sure hidden_dim matches what your model expects
    model = HybridEquivariantGNN(hidden_dim=16)
    model.eval()
    
    # 4. Get Prediction for Original
    with torch.no_grad():
        out_original = model(data_original)
    
    # 5. Rotate the Molecule
    rot_matrix = get_random_rotation_matrix()
    pos_rotated = torch.matmul(pos, rot_matrix)
    
    # Create rotated data (keep x and edge_attr the same!)
    data_rotated = Data(x=x, pos=pos_rotated, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # 6. Get Prediction for Rotated
    with torch.no_grad():
        out_rotated = model(data_rotated)
    
    # 7. Compare
    diff = torch.abs(out_original - out_rotated).item()
    print(f"Original Output: {out_original.item():.6f}")
    print(f"Rotated Output:  {out_rotated.item():.6f}")
    print(f"Difference:      {diff:.6e}")
    
    # Allow for small floating point errors (1e-5 is standard tolerance)
    if diff < 1e-4:
        print("✅ SUCCESS: Model is Equivariant! (Physics Preserved)")
    else:
        print("❌ FAILURE: Model is sensitive to rotation.")

if __name__ == "__main__":
    test_equivariance()
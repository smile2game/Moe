import numpy as np

class Shard:
    def __init__(self, dim):
        self.dim = dim
    
    def is_shard(self):
        return True
    
    def get_dim(self):
        return self.dim

class Replicate:
    def __init__(self):
        pass
    
    def is_shard(self):
        return False

class Partial:
    def __init__(self, reduce_type='sum'):
        self.reduce_type = reduce_type
    
    def is_shard(self):
        return False
    
    def is_partial(self):
        return True

class ProcessMesh:
    def __init__(self, shape, process_ids, dim_names):
        self.shape = shape
        self.process_ids = process_ids
        self.dim_names = dim_names

class Tensor:
    def __init__(self, shape, process_mesh, placements):
        self.shape = shape
        self.process_mesh = process_mesh
        self.placements = placements

def shard_submesh_and_slice(sub_mesh, tensor_slice, tensor_dim, mesh_dim, global_mesh_shape):
    """Progressively shard sub-meshes and tensor slices."""
    num_shards = global_mesh_shape[mesh_dim]
    total_size = tensor_slice[tensor_dim][1] - tensor_slice[tensor_dim][0]
    shard_size = total_size // num_shards
    
    if total_size % num_shards != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} size {total_size} cannot be evenly divided by {num_shards}")
    
    sub_mesh_size = len(sub_mesh)
    if sub_mesh_size % num_shards != 0:
        raise ValueError(f"Sub-mesh size {sub_mesh_size} cannot be evenly divided by {num_shards}")
    shard_mesh_size = sub_mesh_size // num_shards
    
    new_sub_meshes = []
    new_slices = []
    
    for i in range(num_shards):
        start_idx = i * shard_mesh_size
        end_idx = (i + 1) * shard_mesh_size
        new_sub_mesh = tuple(sub_mesh[start_idx:end_idx])
        new_sub_meshes.append(new_sub_mesh)
        
        new_slice = list(tensor_slice)
        start = tensor_slice[tensor_dim][0] + i * shard_size
        end = start + shard_size
        new_slice[tensor_dim] = (int(start), int(end))
        new_slices.append(tuple(new_slice))
    
    return new_sub_meshes, new_slices

def get_local_slices(tensor, mesh, placements):
    """Compute local tensor slices, supporting Shard, Replicate, and Partial."""
    if len(mesh.shape) != len(placements):
        raise ValueError(f"Number of placements ({len(placements)}) must match mesh dimensions ({len(mesh.shape)})")
    
    sub_mesh2tensor_indices = {
        tuple(mesh.process_ids): {
            'slice': [(0, s) for s in tensor.shape],
            'partial': {}
        }
    }
    
    print("---------------------------- Initial State ----------------------------")
    for sub_mesh, info in sub_mesh2tensor_indices.items():
        print(f"Process Group {sub_mesh}: Slice {info['slice']}, Partial Info {info['partial']}")
    print("-----------------------------------------------------------------")
    
    for mesh_dim, placement in enumerate(placements):
        new_sub_mesh2tensor_indices = {}
        
        if placement.is_shard():
            tensor_dim = placement.get_dim()
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_sub_meshes, new_slices = shard_submesh_and_slice(
                    sub_mesh, info['slice'], tensor_dim, mesh_dim, mesh.shape
                )
                for new_sub_mesh, new_slice in zip(new_sub_meshes, new_slices):
                    new_sub_mesh2tensor_indices[new_sub_mesh] = {
                        'slice': new_slice,
                        'partial': info['partial'].copy()
                    }
        elif hasattr(placement, 'is_partial') and placement.is_partial():
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_partial = info['partial'].copy()
                new_partial[mesh_dim] = placement.reduce_type
                new_sub_mesh2tensor_indices[sub_mesh] = {
                    'slice': info['slice'],
                    'partial': new_partial
                }
        else:  # Replicate
            new_sub_mesh2tensor_indices = sub_mesh2tensor_indices.copy()
        
        sub_mesh2tensor_indices = new_sub_mesh2tensor_indices
        
        print(f"---------------------------- Processing Placement {mesh_dim}: {type(placement).__name__} ----------------------------")
        for sub_mesh, info in sub_mesh2tensor_indices.items():
            print(f"Process Group {sub_mesh}: Slice {info['slice']}, Partial Info {info['partial']}")
        print("-----------------------------------------------------------------")
    
    return sub_mesh2tensor_indices

def get_rank2tensor_indices(sub_mesh2tensor_indices):
    """Convert sub_mesh2tensor_indices to rank2tensor_indices."""
    rank2tensor_indices = {}
    for process_group, info in sub_mesh2tensor_indices.items():
        for rank in process_group:
            rank2tensor_indices[rank] = {
                'slice': info['slice'],
                'partial': info['partial']
            }
    return rank2tensor_indices

# Test Examples
if __name__ == "__main__":
    # Example 1: 2D Tensor, 2D Mesh, Two Shard(0)
    dist_tensor = Tensor(
        shape=[4, 4],
        process_mesh=ProcessMesh(
            shape=[2, 2],
            process_ids=[0, 1, 2, 3],
            dim_names=['x', 'y']
        ),
        placements=[Shard(0), Shard(0)]
    )
    print("---------------------------- Example - Two Shard(0) ----------------------------")
    sub_mesh2tensor_indices = get_local_slices(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements)
    
    print("\nFinal sub_mesh2tensor_indices:")
    for process_ids, info in sub_mesh2tensor_indices.items():
        print(f"Process Group {process_ids}: Slice {info['slice']}, Partial Info {info['partial']}")
    
    rank2tensor_indices = get_rank2tensor_indices(sub_mesh2tensor_indices)
    print("\n---------------------------- rank2tensor_indices ----------------------------")
    for rank, info in rank2tensor_indices.items():
        print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")
    
    # Example 2: Including Partial with Corrected Mesh
    dist_tensor_partial = Tensor(
        shape=[4, 4],
        process_mesh=ProcessMesh(
            shape=[1, 2, 2, 1],
            process_ids=[0, 1, 2, 3],
            dim_names=['x', 'y', 'z', 't']
        ),
        placements=[Replicate(), Shard(1), Replicate(), Partial()]
    )
    print("\n---------------------------- Example - Including Partial ----------------------------")
    sub_mesh2tensor_indices_partial = get_local_slices(dist_tensor_partial, dist_tensor_partial.process_mesh, dist_tensor_partial.placements)
    
    print("\nFinal sub_mesh2tensor_indices:")
    for process_ids, info in sub_mesh2tensor_indices_partial.items():
        print(f"Process Group {process_ids}: Slice {info['slice']}, Partial Info {info['partial']}")
    
    rank2tensor_indices_partial = get_rank2tensor_indices(sub_mesh2tensor_indices_partial)
    print("\n---------------------------- rank2tensor_indices ----------------------------")
    for rank, info in rank2tensor_indices_partial.items():
        print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")
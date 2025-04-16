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

def get_sub_meshes_for_shard(mesh, mesh_dim):
    """为 Shard placement 生成子网格 ProcessMesh 对象"""
    process_ids = np.array(mesh.process_ids).reshape(mesh.shape)
    num_shards = mesh.shape[mesh_dim]
    
    sub_meshes = []
    for i in range(num_shards):
        coords = [slice(None)] * len(mesh.shape)
        coords[mesh_dim] = i
        sub_process_ids = process_ids[tuple(coords)].flatten().tolist()
        new_shape = list(mesh.shape)
        new_shape[mesh_dim] = 1
        sub_mesh = ProcessMesh(new_shape, sub_process_ids, mesh.dim_names)
        sub_meshes.append(sub_mesh)
    return sub_meshes

def shard_submesh_and_slice(mesh, tensor_slice, tensor_dim, mesh_dim): 
    new_sub_meshes = get_sub_meshes_for_shard(mesh, mesh_dim)
    num_shards = len(new_sub_meshes)
    
    total_size = tensor_slice[tensor_dim][1] - tensor_slice[tensor_dim][0]
    shard_size = (total_size + num_shards - 1) // num_shards
    effective_size = shard_size * (num_shards - 1)
    last_shard_size = total_size - effective_size
    
    if total_size % num_shards != 0:
        print(f"警告: 张量维度 {tensor_dim} 的大小 {total_size} 不能被 {num_shards} 整除，将均匀切分并截断余数。")

    new_slices = []
    for i in range(num_shards):
        start = tensor_slice[tensor_dim][0] + i * shard_size
        if i == num_shards - 1:
            end = min(start + last_shard_size, tensor_slice[tensor_dim][1])
        else:
            end = min(start + shard_size, tensor_slice[tensor_dim][1])
        new_slice = list(tensor_slice)
        new_slice[tensor_dim] = (start, end)
        new_slices.append(tuple(new_slice))
    return new_sub_meshes, new_slices

def get_local_slices(tensor, mesh, placements):
    if len(mesh.shape) != len(placements):
        raise ValueError(f"placements 的数量 ({len(placements)}) 必须与网格维度 ({len(mesh.shape)}) 匹配")
    
    sub_mesh2tensor_indices = {
        mesh: {
            'slice': [(0, s) for s in tensor.shape],
            'partial': {}
        }
    }
    print("---------------------------- Initial State ----------------------------")
    for sub_mesh, info in sub_mesh2tensor_indices.items():
        print(f"ProcessMesh {sub_mesh.process_ids} (shape {sub_mesh.shape}): Slice {info['slice']}, Partial Info {info['partial']}\n")

    for mesh_dim, placement in enumerate(placements):
        new_sub_mesh2tensor_indices = {}
        
        if placement.is_shard():
            tensor_dim = placement.get_dim()

            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_sub_meshes, new_slices = shard_submesh_and_slice(
                    sub_mesh, info['slice'], tensor_dim, mesh_dim
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
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_sub_mesh2tensor_indices[sub_mesh] = info.copy()
        
        sub_mesh2tensor_indices = new_sub_mesh2tensor_indices
        print(f"---------------------------- Processing Placement {mesh_dim}: {type(placement).__name__} ----------------------------")
        for sub_mesh, info in sub_mesh2tensor_indices.items():
            print(f"ProcessMesh {sub_mesh.process_ids} (shape {sub_mesh.shape}): Slice {info['slice']}, Partial Info {info['partial']}")
        print("-----------------------------------------------------------------")
    return sub_mesh2tensor_indices

def get_rank2tensor_indices(sub_mesh2tensor_indices):
    rank2tensor_indices = {}
    for sub_mesh, info in sub_mesh2tensor_indices.items():
        for rank in sub_mesh.process_ids:
            rank2tensor_indices[rank] = {
                'slice': info['slice'],
                'partial': info['partial']
            }
    return rank2tensor_indices

if __name__ == "__main__":
    dist_tensor = Tensor(
        shape=[4, 4],
        process_mesh=ProcessMesh(
            shape=[2, 3],
            process_ids=[0, 1, 2, 3, 4, 5],
            dim_names=['x', 'y']
        ),
        placements=[Replicate(), Shard(1)]
    )
    sub_mesh2tensor_indices = get_local_slices(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements)
    
    print("\nFinal sub_mesh2tensor_indices:")
    for sub_mesh, info in sub_mesh2tensor_indices.items():
        print(f"ProcessMesh {sub_mesh.process_ids} (shape {sub_mesh.shape}): Slice {info['slice']}, Partial Info {info['partial']}")
    
    rank2tensor_indices = get_rank2tensor_indices(sub_mesh2tensor_indices)
    print("\n---------------------------- rank2tensor_indices ----------------------------")
    for rank, info in rank2tensor_indices.items():
        print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")
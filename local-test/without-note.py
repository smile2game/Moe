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

def get_sub_meshes_for_shard(sub_mesh, mesh_dim, global_mesh_shape):
    process_ids = np.array(sub_mesh).reshape(global_mesh_shape)
    num_shards = global_mesh_shape[mesh_dim]
    sub_meshes = []
    for i in range(num_shards):
        coords = [slice(None)] * len(global_mesh_shape)
        coords[mesh_dim] = i
        sub_process_ids = process_ids[tuple(coords)].flatten().tolist()
        sub_meshes.append(tuple(sub_process_ids))
    return sub_meshes

def shard_submesh_and_slice(sub_mesh, tensor_slice, tensor_dim, mesh_dim, global_mesh_shape):
    new_sub_meshes = get_sub_meshes_for_shard(sub_mesh, mesh_dim, global_mesh_shape)
    num_shards = len(new_sub_meshes)
    total_size = tensor_slice[tensor_dim][1] - tensor_slice[tensor_dim][0]
    shard_size = total_size // num_shards
    if total_size % num_shards != 0:
        raise ValueError(f"Tensor 维度 {tensor_dim} 的大小 {total_size} 不能被 {num_shards} 整除")
    new_slices = []
    for i in range(num_shards):
        start = tensor_slice[tensor_dim][0] + i * shard_size
        end = start + shard_size
        new_slice = list(tensor_slice)
        new_slice[tensor_dim] = (start, end)
        new_slices.append(tuple(new_slice))
    return new_sub_meshes, new_slices

def get_local_slices(tensor, mesh, placements):
    if len(mesh.shape) != len(placements):
        raise ValueError(f"placements 的数量 ({len(placements)}) 必须与网格维度 ({len(mesh.shape)}) 匹配")
    sub_mesh2tensor_indices = {
        tuple(mesh.process_ids): {
            'slice': [(0, s) for s in tensor.shape],
            'partial': {},
            'mesh_shape': list(mesh.shape)
        }
    }
    for mesh_dim, placement in enumerate(placements):
        new_sub_mesh2tensor_indices = {}
        if placement.is_shard():
            tensor_dim = placement.get_dim()
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_sub_meshes, new_slices = shard_submesh_and_slice(
                    sub_mesh, info['slice'], tensor_dim, mesh_dim, info['mesh_shape']
                )
                for new_sub_mesh, new_slice in zip(new_sub_meshes, new_slices):
                    new_mesh_shape = info['mesh_shape'].copy()
                    new_mesh_shape[mesh_dim] = 1
                    new_sub_mesh2tensor_indices[new_sub_mesh] = {
                        'slice': new_slice,
                        'partial': info['partial'].copy(),
                        'mesh_shape': new_mesh_shape
                    }
        elif hasattr(placement, 'is_partial') and placement.is_partial():
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_partial = info['partial'].copy()
                new_partial[mesh_dim] = placement.reduce_type
                new_sub_mesh2tensor_indices[sub_mesh] = {
                    'slice': info['slice'],
                    'partial': new_partial,
                    'mesh_shape': info['mesh_shape'].copy()
                }
        else:
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_sub_mesh2tensor_indices[sub_mesh] = info.copy()
        sub_mesh2tensor_indices = new_sub_mesh2tensor_indices
    return sub_mesh2tensor_indices

def get_rank2tensor_indices(sub_mesh2tensor_indices):
    rank2tensor_indices = {}
    for process_group, info in sub_mesh2tensor_indices.items():
        for rank in process_group:
            rank2tensor_indices[rank] = {
                'slice': info['slice'],
                'partial': info['partial']
            }
    return rank2tensor_indices

if __name__ == "__main__":
    dist_tensor = Tensor(
        shape=[4, 4],
        process_mesh=ProcessMesh(
            shape=[2, 2, 2],
            process_ids=[0, 1, 2, 3, 4, 5, 6, 7],
            dim_names=['x', 'y', 'z']
        ),
        placements=[Replicate(), Shard(0), Shard(0)]
    )
    sub_mesh2tensor_indices = get_local_slices(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements)

    print("\nFinal sub_mesh2tensor_indices:")
    for process_ids, info in sub_mesh2tensor_indices.items():
        print(f"Process Group {process_ids}: Slice {info['slice']}, Partial Info {info['partial']}")

    rank2tensor_indices = get_rank2tensor_indices(sub_mesh2tensor_indices)
    print("\n---------------------------- rank2tensor_indices ----------------------------")
    for rank, info in rank2tensor_indices.items():
        print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")
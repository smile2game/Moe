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
    # 将 sub_mesh 重塑为网格形状
    process_ids = np.array(sub_mesh).reshape(global_mesh_shape)
    num_shards = global_mesh_shape[mesh_dim]
    
    sub_meshes = []
    for i in range(num_shards):
        # 按 mesh_dim 分片
        coords = [slice(None)] * len(global_mesh_shape)
        coords[mesh_dim] = i
        sub_process_ids = process_ids[tuple(coords)].flatten().tolist()
        print(f"求sub_mesh,coords is {coords},sub_process_ids is {sub_process_ids}")
        sub_meshes.append(tuple(sub_process_ids))
    return sub_meshes

def shard_submesh_and_slice(sub_mesh, tensor_slice, tensor_dim, mesh_dim, global_mesh_shape):
    # 获取分片后的 sub_meshes
    new_sub_meshes = get_sub_meshes_for_shard(sub_mesh, mesh_dim, global_mesh_shape)
    num_shards = len(new_sub_meshes)
    
    # 分片 tensor 维度
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
    """计算本地 tensor slices，支持 Shard、Replicate 和 Partial。"""
    if len(mesh.shape) != len(placements):
        raise ValueError(f"placements 的数量 ({len(placements)}) 必须与网格维度 ({len(mesh.shape)}) 匹配")
    
    # 初始化，包含 mesh_shape
    sub_mesh2tensor_indices = {
        tuple(mesh.process_ids): {
            'slice': [(0, s) for s in tensor.shape],
            'partial': {},
            'mesh_shape': list(mesh.shape)  # 跟踪网格形状
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
                    sub_mesh, info['slice'], tensor_dim, mesh_dim, info['mesh_shape']
                )
                for new_sub_mesh, new_slice in zip(new_sub_meshes, new_slices):
                    # 更新 mesh_shape
                    new_mesh_shape = info['mesh_shape'].copy()
                    new_mesh_shape[mesh_dim] = 1  # 分片后此维度大小变为 1
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
        else:  # Replicate
            for sub_mesh, info in sub_mesh2tensor_indices.items():
                new_sub_mesh2tensor_indices[sub_mesh] = info.copy()
        
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
    #origin
    # dist_tensor = Tensor(
    # shape=[4, 4],
    # process_mesh=ProcessMesh(
    #     shape=[2, 2],
    #     process_ids=[0, 1, 2, 3],
    #     dim_names=['x', 'y']
    # ),
    # # placements=[Replicate(),Shard(0)]
    # placements=[Partial(),Shard(0)]
    # # placements=[Shard(1),Shard(1)]

    # )

    # print("---------------------------- Example - Two Shard(0) ----------------------------")
    # sub_mesh2tensor_indices = get_local_slices(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements)

    # print("\nFinal sub_mesh2tensor_indices:")
    # for process_ids, info in sub_mesh2tensor_indices.items():
    #     print(f"Process Group {process_ids}: Slice {info['slice']}, Partial Info {info['partial']}")
    
    # rank2tensor_indices = get_rank2tensor_indices(sub_mesh2tensor_indices)
    # print("\n---------------------------- rank2tensor_indices ----------------------------")
    # for rank, info in rank2tensor_indices.items():
    #     print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")


    # Example 1: 2D Tensor, 2D Mesh, Two Shard(0)
    dist_tensor = Tensor(
        shape=[4, 4],
        process_mesh=ProcessMesh(
            shape=[2, 2,2],
            process_ids=[0, 1, 2, 3,4,5,6,7],
            dim_names=['x', 'y','z']
        ),
        placements=[Replicate(),Shard(0),Shard(0)]
    )
    """
    >>origin
    [ [[0, 1]z轴, [2, 3]]y轴 , [[4, 5], [6, 7]] ]x轴

    >>shard(0),y轴切分
    [[0,1]z轴 [4,5] ]x轴
    [[2,3]z轴 [6,7] ]x轴

    >>shard(0),z轴切分
    [0,4]x轴
    [1,5]x轴
    [2,6]x轴
    [3,7]x轴
    """

    print("---------------------------- Example - Two Shard(0) ----------------------------")
    sub_mesh2tensor_indices = get_local_slices(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements)


    
    # print("\nFinal sub_mesh2tensor_indices:")
    # for process_ids, info in sub_mesh2tensor_indices.items():
    #     print(f"Process Group {process_ids}: Slice {info['slice']}, Partial Info {info['partial']}")
    
    # rank2tensor_indices = get_rank2tensor_indices(sub_mesh2tensor_indices)
    # print("\n---------------------------- rank2tensor_indices ----------------------------")
    # for rank, info in rank2tensor_indices.items():
    #     print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")
    



    # # Example 2: Including Partial with Corrected Mesh
    # dist_tensor_partial = Tensor(
    #     shape=[4, 4],
    #     process_mesh=ProcessMesh(
    #         shape=[1, 2, 2, 1],
    #         process_ids=[0, 1, 2, 3],
    #         dim_names=['x', 'y', 'z', 't']
    #     ),
    #     placements=[Replicate(), Shard(1), Replicate(), Partial()]
    # )
    # print("\n---------------------------- Example - Including Partial ----------------------------")
    # sub_mesh2tensor_indices_partial = get_local_slices(dist_tensor_partial, dist_tensor_partial.process_mesh, dist_tensor_partial.placements)
    
    # print("\nFinal sub_mesh2tensor_indices:")
    # for process_ids, info in sub_mesh2tensor_indices_partial.items():
    #     print(f"Process Group {process_ids}: Slice {info['slice']}, Partial Info {info['partial']}")
    
    # rank2tensor_indices_partial = get_rank2tensor_indices(sub_mesh2tensor_indices_partial)
    # print("\n---------------------------- rank2tensor_indices ----------------------------")
    # for rank, info in rank2tensor_indices_partial.items():
    #     print(f"Rank {rank}: Slice {info['slice']}, Partial Info {info['partial']}")
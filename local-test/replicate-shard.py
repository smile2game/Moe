class Shard:
    def __init__(self, dim):
        self.dim = dim
    
    def is_shard(self):
        return True
    
    def get_dim(self):
        return self.dim

# 新增 Replicate 类
class Replicate:
    def __init__(self):
        pass
    
    def is_shard(self):
        return False

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




from itertools import product

def get_process_ids_from_coords(mesh, coords):
    """根据坐标列表计算进程 ID"""
    import numpy as np
    process_ids = np.array(mesh.process_ids).reshape(mesh.shape)
    indices = tuple(coords)
    return process_ids[indices].flatten().tolist()

def get_local_slices(tensor, mesh, placements):
    """
    计算每个子网格的本地张量切片。
    
    Args:
        tensor: 输入张量，具有 shape 属性
        mesh: 进程网格，具有 shape 和 process_ids 属性
        placements: 放置策略列表，长度等于网格维度数，每个元素是 Shard 或 Replicate
    
    Returns:
        dict: 子网格进程 ID 元组到张量切片的映射
    """
    D = len(mesh.shape)
    
    # 确定 Shard 和 Replicate 维度
    shard_dims = [d for d in range(D) if placements[d].is_shard()]
    replicate_dims = [d for d in range(D) if not placements[d].is_shard()]
    
    # 获取 Shard 维度的所有坐标组合
    shard_coords_combinations = list(product(*[range(mesh.shape[d]) for d in shard_dims]))
    
    sub_mesh2tensor_indices = {}
    for shard_coords in shard_coords_combinations:
        # 构建 fixed_coords：Shard 维度固定，Replicate 维度取所有坐标
        fixed_coords = [None] * D
        for d in replicate_dims:
            fixed_coords[d] = list(range(mesh.shape[d]))
        for idx, d in enumerate(shard_dims):
            fixed_coords[d] = [shard_coords[idx]]
        
        # 计算子网格的进程 ID
        process_ids = tuple(get_process_ids_from_coords(mesh, fixed_coords))
        
        # 计算张量切片
        tensor_slice = [(0, s) for s in tensor.shape]
        for idx, mesh_dim in enumerate(shard_dims):
            tensor_dim = placements[mesh_dim].get_dim()  # Shard 对应的张量维度
            num_shards = mesh.shape[mesh_dim]
            total_size = tensor.shape[tensor_dim]
            shard_size = total_size // num_shards
            if total_size % num_shards != 0:
                raise ValueError(f"张量维度 {tensor_dim} 的大小 {total_size} 不能被 {num_shards} 整除")
            coord = shard_coords[idx]
            start = coord * shard_size
            end = start + shard_size
            tensor_slice[tensor_dim] = (start, end)
            
        sub_mesh2tensor_indices[process_ids] = tensor_slice
    return sub_mesh2tensor_indices

# 测试示例
import math
if __name__ == "__main__":
    # 示例 2：3D 张量，3D 网格，3 次切分
    dist_tensor2 = Tensor(
        shape=[4,4],
        process_mesh=ProcessMesh(
            shape=[2, 2],
            process_ids=list(range(math.prod([2, 2]))),
            dim_names=['x', 'y']
        ),
        placements=[Replicate(), Shard(1)]  # 沿第 0、1、2 维切分
    )
    print("----------------------------示例 2 - 3D 张量，3D 网格，3 次切分:---------------------------")
    sub_mesh2tensor_indices2 = get_local_slices(dist_tensor2, dist_tensor2.process_mesh, dist_tensor2.placements)
    print(f"sub_mesh2tensor_indices is {sub_mesh2tensor_indices2}")

    # 新增示例 3：3D 张量，3D 网格，包含 Replicate
    # dist_tensor3 = Tensor(
    #     shape=[6, 6, 6],
    #     process_mesh=ProcessMesh(
    #         shape=[2, 2, 2],
    #         process_ids=list(range(math.prod([2, 2, 2]))),
    #         dim_names=['x', 'y', 'z']
    #     ),
    #     placements=[Shard(0), Shard(1), Shard(2)]  # 沿第 0 维切分，复制，沿第 2 维切分
    # )
    # print("----------------------------示例 3 - 3D 张量，3D 网格，包含 Replicate:---------------------------")
    # sub_mesh2tensor_indices3 = get_local_slices(dist_tensor3, dist_tensor3.process_mesh, dist_tensor3.placements)
    # print(f"sub_mesh2tensor_indices is {sub_mesh2tensor_indices3}")
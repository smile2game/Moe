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

def shard_submesh_and_slice(sub_mesh, tensor_slice, tensor_dim, mesh_dim, global_mesh_shape):
    """沿指定网格维度切分 sub_mesh 和对应的张量切片
    Args:
        sub_mesh: 当前子网格的进程 ID 元组
        tensor_slice: 当前张量的切片，格式为 [(start1, end1), (start2, end2), ...]
        tensor_dim: 张量的切分维度
        mesh_dim: 网格的切分维度
        global_mesh_shape: 全局网格的形状
    Returns:
        new_sub_meshes: 切分后的子网格列表
        new_slices: 切分后的张量切片列表
    """
    num_shards = global_mesh_shape[mesh_dim]  # 根据全局网格决定有几个 shards
    total_size = tensor_slice[tensor_dim][1] - tensor_slice[tensor_dim][0]
    shard_size = total_size // num_shards  # 使用整除
    
    # 确保 total_size 能被 num_shards 整除
    if total_size % num_shards != 0:
        raise ValueError(f"Tensor dimension {tensor_dim} size {total_size} cannot be evenly divided by {num_shards}")
    
    # 将 sub_mesh 转换为数组并重塑为网格形状的一部分
    new_sub_meshes = []
    new_slices = []

    for i in range(num_shards):
        # 简单假设：沿 mesh_dim 维度平均切分子网格
        start_idx = i * (len(sub_mesh) // num_shards)
        end_idx = (i + 1) * (len(sub_mesh) // num_shards)
        new_sub_mesh = tuple(sub_mesh[start_idx:end_idx])
        new_sub_meshes.append(new_sub_mesh)

        # 切分张量切片
        new_slice = list(tensor_slice)
        start = tensor_slice[tensor_dim][0] + i * shard_size
        end = start + shard_size
        new_slice[tensor_dim] = (int(start), int(end))  # 强制转换为整数
        new_slices.append(tuple(new_slice))
    return new_sub_meshes, new_slices

def get_local_slices(tensor, mesh, placements):
    """计算每个 rank 的本地张量切片
    Args:
        tensor: 张量实例
        mesh: 处理器网格实例
        placements: 切分策略列表
    Returns:
        rank_slices: 字典，键为 rank ID，值为对应的张量切片
    """
    sub_mesh2tensor_indices = {}
    # 初始化：将整个张量分配给全局网格
    initial_sub_mesh = tuple(mesh.process_ids)  # 例如 (0,1,2,3)
    sub_mesh2tensor_indices[initial_sub_mesh] = [(0, s) for s in tensor.shape]

    # 迭代处理每个 placement
    for mesh_dim, placement in enumerate(placements):
        if placement.is_shard():  # 只有当 placement 是 Shard 时才切分
            tensor_dim = placement.get_dim()  # 张量的切分维度
            new_sub_mesh2tensor_indices = {}
            # 对当前所有的 sub_mesh 进行切分
            for sub_mesh, tensor_slice in sub_mesh2tensor_indices.items():
                # 切分
                new_sub_meshes, new_slices = shard_submesh_and_slice(
                    sub_mesh, tensor_slice, tensor_dim, mesh_dim, mesh.shape
                )
                # 更新
                for new_sub_mesh, new_slice in zip(new_sub_meshes, new_slices):
                    new_sub_mesh2tensor_indices[new_sub_mesh] = new_slice
            sub_mesh2tensor_indices = new_sub_mesh2tensor_indices
        # 如果是 Replicate，则跳过切分，保持 sub_mesh2tensor_indices 不变
        print(f"第{mesh_dim}刀, sub_mesh2tensor_indices is {sub_mesh2tensor_indices}")
    return sub_mesh2tensor_indices

# 测试示例
import math
if __name__ == "__main__":
    # # 示例 2：3D 张量，3D 网格，3 次切分
    # dist_tensor2 = Tensor(
    #     shape=[6, 6, 6],
    #     process_mesh=ProcessMesh(
    #         shape=[2, 3, 1],
    #         process_ids=list(range(math.prod([2, 3, 1]))),
    #         dim_names=['x', 'y', 'z']
    #     ),
    #     placements=[Shard(0), Shard(1), Shard(2)]  # 沿第 0、1、2 维切分
    # )
    # print("----------------------------示例 2 - 3D 张量，3D 网格，3 次切分:---------------------------")
    # sub_mesh2tensor_indices2 = get_local_slices(dist_tensor2, dist_tensor2.process_mesh, dist_tensor2.placements)
    # print(f"sub_mesh2tensor_indices is {sub_mesh2tensor_indices2}")

    # 新增示例 3：3D 张量，3D 网格，包含 Replicate
    dist_tensor3 = Tensor(
        shape=[6, 6, 6],
        process_mesh=ProcessMesh(
            shape=[2, 3, 1],
            process_ids=list(range(math.prod([2, 3, 1]))),
            dim_names=['x', 'y', 'z']
        ),
        placements=[Shard(0), Replicate(), Shard(2)]  # 沿第 0 维切分，复制，沿第 2 维切分
    )
    print("----------------------------示例 3 - 3D 张量，3D 网格，包含 Replicate:---------------------------")
    sub_mesh2tensor_indices3 = get_local_slices(dist_tensor3, dist_tensor3.process_mesh, dist_tensor3.placements)
    print(f"sub_mesh2tensor_indices is {sub_mesh2tensor_indices3}")
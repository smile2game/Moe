def compute_linear_index(coords, shape):
    """将多维坐标转换为线性索引（行优先顺序）
    
    Args:
        coords: 坐标列表，例如 [1, 1]
        shape: 形状列表，例如 [2, 2]
    
    Returns:
        int: 线性索引，例如 3
    """
    index = 0
    stride = 1
    for c, s in zip(coords, shape):
        index += c * stride
        stride *= s
    return index

def get_coords_from_rank(mesh_shape, rank):
    """将 rank 转换为网格中的多维坐标
    
    Args:
        mesh_shape: 网格形状，例如 (2, 2)
        rank: 进程编号，例如 3
    
    Returns:
        tuple: 坐标，例如 (1, 1)
    """
    indices = []
    for dim in mesh_shape:
        rank, index = divmod(rank, dim)
        indices.append(index)
    return tuple(indices)

def get_local_slice(dist_tensor, mesh, placements, rank):
    """计算当前 rank 存储的张量本地切片，支持张量维度被多个网格维度分片
    
    Args:
        dist_tensor: 张量对象，包含 shape 属性
        mesh: 网格对象，包含 shape 属性
        placements: 分片策略列表，例如 [Shard(0), Shard(0)]
        rank: 当前进程编号
    
    Returns:
        list: 切片列表，例如 [slice(2, 4), slice(None)]
    """
    tensor_shape = dist_tensor.shape  # 张量形状，例如 [4, 4]
    mesh_shape = mesh.shape          # 网格形状，例如 [2, 2]
    tensor_ndim = len(tensor_shape)  # 张量维度数
    mesh_ndim = len(mesh_shape)      # 网格维度数

    # 步骤 1：构建 shard_map，允许一个张量维度被多个网格维度分片
    shard_map = {}  # {tensor_dim: [mesh_dims]}
    for mesh_dim, placement in enumerate(placements):
        if placement.is_shard():
            tensor_dim = placement.get_dim()
            if tensor_dim >= tensor_ndim:
                raise ValueError(f"张量维度 {tensor_dim} 超出张量实际维度数 {tensor_ndim}")
            if tensor_dim not in shard_map:
                shard_map[tensor_dim] = []
            shard_map[tensor_dim].append(mesh_dim)

    # 步骤 2：获取当前 rank 在网格中的坐标
    coords = get_coords_from_rank(mesh_shape, rank)

    # 步骤 3：为每个张量维度计算分片索引
    slices = []
    for dim in range(tensor_ndim):
        if dim not in shard_map:
            # 未被分片的维度，取整个维度
            slices.append(slice(None))
        else:
            # 该维度被一个或多个网格维度分片
            M = shard_map[dim]  # 分片该维度的网格维度列表
            total_shards = 1
            for m in M: #累乘求得2 mesh_dim对1 tensor_dim的总分片数
                total_shards *= mesh_shape[m]
            
            dim_size = tensor_shape[dim]
            if dim_size % total_shards != 0:
                raise ValueError(f"张量维度 {dim} 的大小 {dim_size} 必须能被总分片数 {total_shards} 整除")
            shard_size = dim_size // total_shards

            # 计算当前 rank 在参与分片的网格维度上的坐标和形状，从而得到是第几个分片
            coords_M = [coords[m] for m in M]
            shape_M = [mesh_shape[m] for m in M]
            k = compute_linear_index(coords_M, shape_M)

            start = k * shard_size
            end = (k + 1) * shard_size
            slices.append(slice(start, end))
    
    return slices

# 定义辅助类（保持不变）
class Shard:
    def __init__(self, dim):
        self.dim = dim
    
    def is_shard(self):
        return True
    
    def get_dim(self):
        return self.dim

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



if __name__ == "__main__":
    # # 原始情况
    # dist_tensor = Tensor(
    #     shape=[4, 4],
    #     process_mesh=ProcessMesh(
    #         shape=[2, 2],
    #         process_ids=[0, 1, 2, 3],
    #         dim_names=['x', 'y']
    #     ),
    #     placements=[Shard(0), Shard(1)]
    # )

    # for rank in range(4):
    #     slices = get_local_slice(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements, rank)
    #     print(f"Rank {rank} 存储的索引: dim0={slices[0]}, dim1={slices[1]}")


    #切两刀情况
    dist_tensor = Tensor(
        shape=[4, 4],
        process_mesh=ProcessMesh(
            shape=[2, 2],
            process_ids=[0, 1, 2, 3],
            dim_names=['x', 'y']
        ),
        placements=[Shard(0), Shard(0)]
    )

    for rank in range(4):
    # rank = 3
        slices = get_local_slice(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements, rank)
        print(f"Rank {rank} 存储的索引: dim0={slices[0]}, dim1={slices[1]}")

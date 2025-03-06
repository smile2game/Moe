def get_coords_from_rank(mesh_shape,rank):
    """ 根据rank顺序,行优先坐标展开,依次求余数，更新缩小rank,得到在mesh_shape中的位置 """
    indices = []  # 用来存储转换后的多维索引
    for dim in mesh_shape:  # 从第一个维度开始处理
        rank,index = divmod(rank,dim) #除法 求得 商和余数
        indices.append(index)  # 获取当前维度的索引
    return tuple(indices)  # 返回元组形式的多维索引


def get_local_slice(dist_tensor, mesh, placements, rank):
    """
    计算给定 rank 上存储的分布式张量的索引范围。

    参数:
        dist_tensor: 分布式张量，包含 shape 属性（list 或 tuple）
        mesh: ProcessMesh，包含 shape 属性（list 或 tuple）
        placements: Placement 列表，长度与网格维度数相等，例如 [Shard(0), Replicate(), ...]
        rank: 当前进程的 rank

    返回:
        list of slice，每个元素是一个 slice 对象，表示该 rank 上张量的索引范围
    """
    tensor_shape = dist_tensor.shape  # 张量形状，例如 [4, 4, 4]
    mesh_shape = mesh.shape          # 网格形状，例如 [2, 2]
    tensor_ndim = len(tensor_shape)  # 张量维度数
    mesh_ndim = len(mesh_shape)      # 网格维度数

    # 步骤 1：解析 placements，构建 {tensor_dim: mesh_dim} 映射
    shard_map = {}  # 记录张量维度沿哪个网格维度分片
    for mesh_dim, placement in enumerate(placements):
        if placement.is_shard():
            tensor_dim = placement.get_dim()
            if tensor_dim >= tensor_ndim:
                raise ValueError(f"张量维度 {tensor_dim} 超出张量实际维度数 {tensor_ndim}")
            if tensor_dim in shard_map: #这个不确定要不要 写出来
                raise ValueError(f"张量维度 {tensor_dim} 被多个网格维度分片，不支持")
            shard_map[tensor_dim] = mesh_dim #tensor_dim -> mesh_dim

    # 步骤 2：获取当前 rank 在网格中的坐标
    coords = get_coords_from_rank(mesh_shape, rank)

    # 步骤 3：为每个张量维度计算分片索引
    slices = []
    for dim in range(tensor_ndim):
        if dim in shard_map:  # 该维度被映射到某个网格维度
            mesh_dim = shard_map[dim]
            m = mesh_shape[mesh_dim]  # 网格维度大小
            if m == 1:
                # 网格维度大小为 1，视为未分片，返回 slice(None)
                slices.append(slice(None)) #如果不加这个判断,返回0:all(会与None冲突)
            else:
                # 网格维度大小大于 1，正常计算分片
                c = coords[mesh_dim]      # 当前进程在该网格维度的坐标
                dim_size = tensor_shape[dim]  # 张量该维度的大小
                if dim_size % m != 0:
                    raise ValueError(f"张量维度 {dim} 的大小 {dim_size} 必须能被网格大小 {m} 整除")
                shard_size = dim_size // m  # 每个分片的大小
                start = c * shard_size #分片的起始索引
                end = (c + 1) * shard_size
                slices.append(slice(start, end)) 
        else:
            # 未分片的维度，保持完整
            slices.append(slice(None))
    return slices

# 定义辅助类
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

# 测试代码

dist_tensor = Tensor(
    shape=[4, 4],  # 3维张量
    process_mesh=ProcessMesh(
        shape=[2],  # 2维网格
        process_ids=[0, 1],
        dim_names=['x']
    ),
    placements=[Shard(0)]  # 第0维沿网格第0维分片，第1维沿网格第1维分片
)

# 总进程数 = 2 * 2 = 4
for rank in range(2):
    slices = get_local_slice(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements, rank)
    print(f"Rank {rank} 存储的索引: {slices[0]},{slices[1]}")
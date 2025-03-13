"""
mesh = [[0,1],[2,3]]  #2x4
placements = [Shard(0), Shard(0)]
tensor = [4,4]  #4x4

第一刀
sub_mesh2tensor_indices = {
                        [0,1]: [(0,2),(0,4)],
                        [2,3]: [(2,4),(0,4)],
                       }


第二刀
sub_mesh2tensor_indices = {
                        [0]: [(0,1),(0,4)],
                        [1]: [(1,2),(0,4)],

                        [2]: [(2,3),(0,4)],
                        [3]: [(3,4),(0,4)],
                       }
"""


def get_slice_on_sub_mesh(tensor, sub_mesh, p, map):
   pass 

def get_1D_sub_process_mesh(process_mesh, mesh_dim):
    """
    Get the 1-D sub process mesh on specific mesh_dim which:
      1) where the reshard should be performed.
      2) contains current process.

    Args:
      process_mesh (ProcessMesh): the global process mesh.
      mesh_dim (int): the mesh dimension where the dist_tensor is
        sharded or partial.

    e.g.
      1) process_mesh = [[0, 1, 2], [3, 4, 5]], axis = 0:
         process rank id      returned sub mesh
           0 or 3               [0, 3]
           1 or 4               [1, 4]
           2 or 5               [2, 5]
      2) process_mesh = [[0, 1, 2], [3, 4, 5]], axis = 1:
         process rank id      returned sub mesh
           0 or 1 or 2          [0, 1, 2]
           3 or 4 or 5          [3, 4, 5]
    """
    import numpy as np

    mesh_shape = process_mesh.shape
    dim_names = process_mesh.dim_names
    process_ids = np.array(process_mesh.process_ids).reshape(mesh_shape)

    rank_id = dist.get_rank()
    # FIXME (JZ-LIANG) Remove this hack to support any op mesh group for Pipeline Parallelism
    if rank_id not in process_mesh.process_ids:
        rank_id = process_mesh.process_ids[0]
    coord = list(np.where(process_ids == rank_id))
    coord[mesh_dim] = range(mesh_shape[mesh_dim])
    sub_process_ids = process_ids[tuple(coord)].flatten()
    sub_mesh_name = dim_names[mesh_dim]

    return dist.ProcessMesh(sub_process_ids, [sub_mesh_name])

def get_sub_mesh(mesh, i):
    sub_mesh_list = []

    return sub_mesh_list

def get_local_slices(tensor, mesh, placements):
    sub_mesh2tensor_indices = {}
    for i, placement in enumerate(placements): #Shard(0) , Shard(0)
        sub_mesh_list = get_sub_mesh(mesh, i)  #i代表这个切是在 mesh的哪个维度
        for sub_mesh in sub_mesh_list:
            get_slice_on_sub_mesh(tensor, sub_mesh, map,placement)

    return rank_slices #{rank0:[(0,1),(0,3)],rank1:[(1,2),(0,3)],rank2:[(2,3),(0,3)] , rank3:[(3,4),(0,3)]}


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

    # for rank in range(4):
    # rank = 3
    rank_slices = get_local_slices(dist_tensor, dist_tensor.process_mesh, dist_tensor.placements) #dim0 = (3,4,None) dim1 = (None,None,None)
    print(f"rank_slices is {rank_slices}")
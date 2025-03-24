import numpy as np
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
        sub_meshes.append(tuple(sub_process_ids))
    
    return sub_meshes


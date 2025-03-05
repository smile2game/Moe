# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np

import paddle
import paddle.distributed as dist


class TestMoEUtils:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seed = eval(os.getenv("seed"))
        self._backend = os.getenv("backend")
        self._mesh0 = dist.ProcessMesh([[0], [1]], dim_names=["x", "y"]) #2x1 
        self._mesh1 = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"]) #1x2
        self._mesh2 = dist.ProcessMesh([0,1],dim_names = ['x']) #一个维度

    def test_local_reshape(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        tgt_shape = [h // 2, w * 2]
        x = paddle.arange(0, h * w).reshape(src_shape)
        x.stop_gradient = False
        np_x = x.numpy()
        print(f"before shard,x is {x}")

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        dist_y = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, [-1, w * 2], self._mesh0, [dist.Shard(1), dist.Replicate()]
        )

        splitted_np_x = np.split(np_x, 2, axis=1)
        for i in range(len(splitted_np_x)):
            splitted_np_x[i] = splitted_np_x[i].reshape([h // 2, w])
        np.testing.assert_array_equal(
            splitted_np_x[dist.get_rank()], dist_y._local_value().numpy()
        )

        label = paddle.ones(tgt_shape, dtype=paddle.int64)
        label.stop_gradient = False
        dist_label = dist.shard_tensor(
            label, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        loss = dist_y - dist_label
        loss.backward()

        np_grad = np.ones(src_shape, dtype="int64")
        splitted_np_grad = np.split(np_grad, 2, axis=1)
        np.testing.assert_array_equal(
            splitted_np_grad[dist.get_rank()],
            dist_x.grad._local_value().numpy(),
        )

        with unittest.TestCase().assertRaises(AssertionError):
            dist_z = dist.auto_parallel.moe_utils._dist_reshape(
                dist_x,
                dist_x.shape,
                self._mesh1,
                [dist.Replicate(), dist.Replicate()],
            )

        # test the warning log message
        dist_z = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, dist_x.shape, self._mesh0, [dist.Shard(1), dist.Shard(1)]
        )

    def test_nd_mesh_alltoall(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        dist_y = dist.reshard(
            dist_x, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        dist_y.backward()

        assert dist_y.placements == [dist.Shard(0), dist.Replicate()]
        assert dist_x.grad.placements == [dist.Shard(1), dist.Replicate()]
        np_grad = np.ones(src_shape, dtype="int64")
        splitted_np_grad = np.split(np_grad, 2, axis=1)
        np.testing.assert_array_equal(
            splitted_np_grad[dist.get_rank()],
            dist_x.grad._local_value().numpy(),
        )

    def test_reshard_mesh_shape_replicate(self):
        """
        1. Replicate 拆分
        [0,1], [Replicate()]  --> [[0],[1]], [Replicate(), Replicate()]
        
        2. Replicate 多层拆分
        [0,1,2,3] [Replicate()]  --> [[[0],[1]],[[2],[3]]]  [Replicate(), Replicate(), Replicate()]
        """
        print("**********replicate_test************")
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        print(f"before shard,\n======================== x is {x}\n========================")
        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        print(f"after shard,\n======================== dist_x is {dist_x}\n========================")
        dist_y = dist.reshard(
            dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )
        print(f"after reshard,\n======================== dist_y is {dist_y}\n========================")    
        assert dist_y.process_mesh == self._mesh1
        print(f"dist_y._local_value().numpy() is {dist_y._local_value().numpy()}\n dist_x._local_value().numpy() is {dist_x._local_value().numpy()}")
        np.testing.assert_array_equal(
            dist_y._local_value().numpy(), dist_x._local_value().numpy()
        )


    def test_reshard_mesh_shape_partial(self):
        """
        1. Partial 合并
        [[0],[1]] [Partial(),Partial()] --> [0,1], [Partial()] 

        2. Partial 多层合并 嵌套 
        [[[0],[1]],[[2],[3]]]  [Partial(), Partial(), Partial()]  --> [0,1,2,3] [Partial()]
        
        

        """
        print("**********partial_test************")
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        print(f"before shard,\n======================== x is {x}\n========================")
        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Partial(), dist.Partial()]
        )
        print(f"after shard,\n======================== dist_x is {dist_x}\n========================")
        dist_y = dist.reshard(
            dist_x, self._mesh1, [dist.Partial()]
        )
        print(f"after reshard,\n======================== dist_y is {dist_y}\n========================")    
        assert dist_y.process_mesh == self._mesh1 #mesh成功改变
        print(f"dist_y._local_value().numpy() is {dist_y._local_value().numpy()}\n dist_x._local_value().numpy() is {dist_x._local_value().numpy()}")
        np.testing.assert_array_equal( #_local_value完全不变
            dist_y._local_value().numpy(), dist_x._local_value().numpy()
        )

    def test_reshard_mesh_shape_shard(self):
        """
        1. Shard维度扩展且冗余
        [0,1], [Shard(0)] 2 --> [[0],[1]], [Shard(0), Shard(1)] 2x1

        [[0],[1]], [Shard(0), Shard(1)] 2x2 --> [0,1], [Shard(0)] 4
        数据分布 4x4
        GPU0: [[0 1 2 3] , [4 5 6 7]]
        GPU1: [[ 8  9 10 11], [12 13 14 15]]

        2. 升维/降维

        3. 跨纬度mesh重排
        [[0,1],[2,3]] 2x2 [Shard(0), Shard(1)] --> [[0,2],[1,3]],2x2 [Shard(1),Shard(0)]

        数据分布:
        GPU0  [[1 2],[5 6]]
        GPU1  [[3 4],[7 8]]
        GPU2  [[9 10],[13 14]]
        GPU3  [[11 12],[15 16]]
        """
        print("**********shard_test************")
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        print(f"before shard,\n======================== x is {x}\n========================")
        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(0), dist.Shard(1)] #[[0,1]] 2x1  
        )
        print(f"dist_x._local_value().numpy() is {dist_x._local_value().numpy()}")

        dist_z = dist.shard_tensor(
            x, self._mesh2, [dist.Shard(0)]
        )
        print(f"dist_z._local_value().numpy() is {dist_z._local_value().numpy()}")
        
        if np.array_equal(dist_x._local_value().numpy(), dist_z._local_value().numpy()):
            print("_local_value不变")
        else:
            print("_local_value改变")

        print(f"shard后,\n======================== dist_x is {dist_x}\n========================")
        dist_y = dist.reshard(
            dist_x, self._mesh2, [dist.Shard(0)]
        )
        print(f"reshard后,\n======================== dist_y is {dist_y}\n========================")    
        assert dist_y.process_mesh == self._mesh2
        print(f"dist_y._local_value().numpy() is {dist_y._local_value().numpy()}\n dist_x._local_value().numpy() is {dist_x._local_value().numpy()}")
        np.testing.assert_array_equal(
            dist_y._local_value().numpy(), dist_x._local_value().numpy()
        )

    def run_test_case(self):
        # self.test_local_reshape()
        # self.test_nd_mesh_alltoall()
        # self.test_reshard_mesh_shape_replicate()
        # self.test_reshard_mesh_shape_partial()
        self.test_reshard_mesh_shape_shard()


if __name__ == '__main__':
    TestMoEUtils().run_test_case() 
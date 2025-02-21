#!/bin/bash

# 创建 .pip 目录
mkdir -p ~/.pip

# 写入 pip.conf 文件
cat > ~/.pip/pip.conf <<EOL
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
EOL

# 提示完成
echo "Pip 阿里云镜像源设置完成!"

#开始设置 paddle
cd bd/Paddle
pip install -r python/requirements.txt

if [ ! -d "build" ]; then
    echo "目录 build 不存在，执行相应操作..."
    # 在这里编写当 build 目录不存在时需要执行的命令
    mkdir build
    echo "已创建 build 目录"
else
    echo "目录 build 已存在"
fi

cd build

# time cmake .. -DPY_VERSION=3.8 -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=ON -DWITH_PYTHON=ON -DWITH_MKL=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DWITH_CINN=OFF -DWITH_PSCORE=OFF -DON_INFER=OFF
#难道这里还需要修改?
# cmake .. \
#   -DPY_VERSION=3.8 \
#   -DWITH_GPU=ON \
#   -DWITH_TESTING=ON \
#   -DWITH_PYTHON=ON \
#   -DWITH_DISTRIBUTED=ON \
#   -DWITH_PIR=ON \
#   -DON_INFER=ON \
#   -DWITH_XPU=OFF \
#   -DCMAKE_BUILD_TYPE=Release



time make -j20 | tee build_log.txt
# pip uninstall paddlepaddle-gpu
pip install /home/aistudio/bd/Paddle/build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
export PYTHONPATH=/home/aistudio/bd/Paddle/test:$PYTHONPATH
#安装torch,用于竞品分析
pip install ipdb

pip install pre-commit==2.17.0
pre-commit install
# pip install torch
# pip install pdbpp
#ipdb更好用



git config --global user.name "smile2game"
git config --global user.email "2426827419@qq.com"
git config --global credential.helper store
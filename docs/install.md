# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n daocc python=3.8 -y
conda activate daocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install [openmpi-4.0.4](https://www.open-mpi.org/software/ompi/v4.0/) (optional and deprecated)**
```shell
# Create an installation path. eg: INSTALL_PATH=~/local/openmpi-4.0.4
tar -xzvf openmpi-4.0.4.tar.gz
cd openmpi-4.0.4
./configure --prefix=INSTALL_PATH
make -j32
make install
# Add the following to your ~/.bashrc or ~/.bash_aliases file.
export PATH=$PATH:INSTALL_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:INSTALL_PATH/lib
```

**d. Install other dependencies.**
```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.20.0
pip install nuscenes-devkit==1.1.11
pip install torchpack==0.3.1
pip install numba==0.48.0
pip install numpy==1.22.0
pip install prettytable==3.11.0
pip install ninja==1.11.1.1
pip install einops==0.8.0
pip install yapf==0.40.0
pip install pillow==8.4.0
pip install mpi4py==3.0.3
pip install Shapely==1.7.1
```

**e. Clone DAOcc and install the codebase.**
```shell
git clone https://github.com/AlphaPlusTT/DAOcc-dev.git
cd DAOcc-dev
python setup.py develop
```

### https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

#-------------------------------------------------------------------------------

### docker pull
IMAGE=nvcr.io/nvidia/pytorch:21.06-py3
docker pull $IMAGE

watch -n0.1 nvidia-smi

#-------------------------------------------------------------------------------

### start docker container
### -u $(id -u):$(id -g) : https://unix.stackexchange.com/a/627028/246969
LOCAL_DIR=/home/xiaofeng.wu/prjs/zip-model/docker_file_system
CONTAINER_DIR=/workspace/outside-docker
CONTAINER_NAME=zip_model
docker run --privileged --name $CONTAINER_NAME --gpus all --ipc=host -it -v $LOCAL_DIR:$CONTAINER_DIR $IMAGE

#-------------------------------------------------------------------------------

### check the zip_model container's status
docker ps -a --filter "name=zip_model"

#-------------------------------------------------------------------------------

### check container OS env
cat /etc/os-release

#-------------------------------------------------------------------------------

### VSCode: Attach to a running container, Developing inside a Container
### https://code.visualstudio.com/docs/devcontainers/attach-container
### open path: /workspace/outside-docker/

### start the exited and attach to it
CONTAINER_NAME=zip_model
docker start $CONTAINER_NAME
docker attach $CONTAINER_NAME

### check all conda env in the docker
conda deactivate
conda info --env
conda activate smoothquant

#-------------------------------------------------------------------------------

### stop a container
CONTAINER_NAME=zip_model
docker stop $CONTAINER_NAME

#-------------------------------------------------------------------------------

### To clean all exited Docker containers, you can use a combination of Docker commands. Here's a step-by-step guide:
docker rm $(docker ps -a -f status=exited -q)

#-------------------------------------------------------------------------------

### setup to use git clone inside a docker
### https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux#generating-a-new-ssh-key
ssh-keygen -t ed25519 -C "shizukanaskytree@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

### https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
cat ~/.ssh/id_ed25519.pub
# Then select and copy the contents of the id_ed25519.pub file
# displayed in the terminal to your clipboard

#-------------------------------------------------------------------------------

apt-get install git-lfs
git lfs install

#-------------------------------------------------------------------------------

### docker 里面有 conda
conda create -n smoothquant python=3.8 -y

### activate conda env
echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
source /opt/conda/etc/profile.d/conda.sh
conda activate smoothquant

#-------------------------------------------------------------------------------

### We implement SmoothQuant INT8 inference for PyTorch with [CUTLASS](https://github.com/NVIDIA/cutlass)
### INT8 GEMM kernels, which are wrapped as PyTorch modules in [torch-int](https://github.com/Guangxuan-Xiao/torch-int).
### Please install [torch-int](https://github.com/Guangxuan-Xiao/torch-int) before running the SmoothQuant PyTorch INT8 inference.
cd ..

### https://github.com/Guangxuan-Xiao/torch-int
# git clone --recurse-submodules https://github.com/Guangxuan-Xiao/torch-int.git ### 只在第一次执行

cd torch-int
conda install -c anaconda gxx_linux-64=9 -y
pip install -r requirements.txt
source environment.sh

### change the compute capacity in build_cutlass.sh
### ONLY THIS LINE: cmake .. -DCUTLASS_NVCC_ARCHS='60;61;70;75;80' -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_UNITY_BUILD_ENABLED=ON # nvcc warning : The 'compute_35', 'compute_37', 'compute_50', 'sm_35', 'sm_37' and 'sm_50' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
nvidia-smi --query-gpu=compute_cap --format=csv
bash build_cutlass.sh

### 缺文件，解法：(1) copy-paste two missing files from your local specific folder OR (2) reinstall cuda driver on the docker
### /usr/local/cuda/lib64/libcublasLt_static.a
### /usr/local/cuda/lib64/libcublas_static.a
python setup.py install

### test torch-int.git
python tests/test_linear_modules.py

#-------------------------------------------------------------------------------

cd $CONTAINER_DIR
cd smoothquant-prj
# git clone https://github.com/mit-han-lab/smoothquant ### 只在第一次执行
cd smoothquant

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers accelerate datasets zstandard
pip install huggingface_hub

# python setup.py install ### if I want to debug the src
pip install -e .

#-------------------------------------------------------------------------------

### vscode -> remote explorer -> Dev Containers -> click zip_model's new window.

#-------------------------------------------------------------------------------

### test ipynb in smoothquant
cd ../smoothquant/examples

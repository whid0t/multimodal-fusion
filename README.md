## Requirements
For DeepViewAgg framework and WildScenes part of thesis:
- Python 3.8
- Pytorch 1.7.1
- Cuda 11.0

For training DeepLabV3+ for the KITTI-360 part:
- Python 3.8
- Pytorch 1.12
- Cuda 11.3

## Installation - KITTI-360
The first part of the thesis (comparison on KITTI-360) was conducted in the Jupyter server of the University of Twente, using a GPU server with an A10 GPU with 24GB VRAM. Installation steps are adapted for the Jupyter sever from the official ```install.sh``` from the official [DeepViewAgg repository](https://github.com/drprojects/DeepViewAgg/tree/release):

### Create and activate conda environemnt<br>
```shell
conda create -n deepviewagg python=3.8
conda activate deepviewagg
```

### Install Python dependencies<br>
```shell
pip install \
    omegaconf \
    wandb \
    tensorboard \
    plyfile \
    hydra-core==1.1.0 \
    pytorch-metric-learning \
    matplotlib \
    seaborn \
    pykeops==1.4.2 \
    imageio \
    opencv-python \
    pypng \
    git+http://github.com/CSAILVision/semantic-segmentation-pytorch.git@master \
    h5py \
    faiss-gpu==1.6.5
```

### Install PyTorch 1.7.1 and CUDA 11.0<br>
```shell
conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch
# Verify installation, should result into "1.7.1+cu11.0
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Install PyTorch Geometric (compatible with 1.7.1 + cu110)<br>
```shell
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install scipy
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.6.3
```

### Install additional tools
```shell
pip install torchnet gdown
pip install torch-points-kernels==0.6.10 --no-cache-dir
```

### Clone DeepViewAgg repository to use a clean, unmodified copy
```shell
git clone https://github.com/drprojects/DeepViewAgg.git
cd DeepViewAgg
```

### Clone Minkowski Engine
```shell
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd Minkowski
```

### Install sparsehash
Since for installation of Minkowski we need root rights, a work around is to install sparsehash and build it.
```shell
git clone https://github.com/sparsehash/sparsehash.git
cd sparsehash
./configure
make install
cd ..
```

### Compile Minkowski engine
```shell
pip install ninja
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
Test installation:
```shell
# Should print 0.5.4
python -c "import MinkowskiEngine as ME; print(ME.__version__)"
```

### (Optional) JupyterLab + Plotly
```shell
pip install plotly==5.4.0 "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1
```

### Extra utilities
```shell
conda install plotly
conda install googledrivedownloader
pip install open3d
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.1.0
```

### If you need to make a notebook and use the conda environment you also need a jupyter kernel
```shell
python -m ipykernel install --user --name=deepviewagg --display-name "Python (deepviewagg)"
```

## Installation for mmsegmentation for 2D-only model

### Create and activate conda environemnt<br>
```shell
conda create -n mmseg2 python=3.8 -y
conda activate mmseg2
```

### Install PyTorch 1.12 and CUDA 11.3<br>
```shell
conda install -c nvidia -c pytorch pytorch=1.12 torchvision cudatoolkit=11.3 -y
```

### Install MMCV and mmsegmentation
```shell
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmsegmentation==0.30.0
```

### Clone and install mmsegmentation from github
```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .
```

### Install additional tools
```shell
pip install tqdm pandas scipy numpy pillow matplotlib imageio ipykernel
```

### Fixing issues with pretty print
There is an error with pretty print, so a file needs fixing.
```shell
nano /opt/miniconda3/envs/mmseg2/lib/python3.8/site-packages/mmcv/utils/config.py
```
Go into the function "format_dict" and remove the flag ```verify=True``` on the line ```(text, _ = FormatCode(text, style_config=yapf_style, verify=True)```

## Dataset setup
### KITTI-360
For downloading the [KITTI-360 dataset](https://www.cvlibs.net/datasets/kitti-360/), a registration is needed. After registering, your request will be approved and you can go to the download page, where you need to download the following files:
- Fisheye cameras
- 2D perspective - train and val, and test
- Semantics of left and right camera
- Confidence of left and right camera
- Accumulated point clouds for train and val, and test
- Calibrations
- Vehicle poses

The data should be structured like this for DeepViewAgg multimodal and 3D-only model:
```
kitti360mm/
        └── raw/
            ├── data_3d_semantics/
            |   └── 2013_05_28_drive_{{seq:0>4}}_sync/
            |       └── static/
            |           └── {{start_frame:0>10}}_{{end_frame:0>10}}.ply
            ├── data_2d_raw/
            |   └── 2013_05_28_drive_{{seq:0>4}}_sync/
            |       ├── image_{{00|01}}/
            |       |   └── data_rect/
            |       |       └── {{frame:0>10}}.png
            |       └── image_{{02|03}}/
            |           └── data_rgb/
            |               └── {{frame:0>10}}.png
            ├── data_poses/
            |   └── 2013_05_28_drive_{{seq:0>4}}_sync/
            |       ├── poses.txt
            |       └── cam0_to_world.txt   
            └── calibration/
                ├── calib_cam_to_pose.txt
                ├── calib_cam_to_velo.txt
                ├── calib_sick_to_velo.txt
                ├── perspective.txt
                └── image_{{02|03}}.yaml
```
For the 2D-only model more freedom is provided.

## Training - KITTI-360

### 3D-only training
For the 3D-only training a script is provided inside ```/DeepViewAgg/train-3donly.sh```. The current configuration is what was used in the thesis and sucessfully completed in 90 hours. You can change the ```DATA_ROOT``` parameter if needed. Run training with ```bash train-3donly.sh```.

### 2D-only training
For the 2D only training the data for 2d was moved into ```DeepViewAgg/dataset/2d```. So the folder contained the raw 2D data, semantics 2d and the corresponding confidence for the labels. Training was done with the files ```mmsegmentation_kitti-360/configs/kitti360/deeplabv3plus_r18-d8_512x1024_20k_k360.py``` (which is the model configuration) and ```mmsegmentation_kitti-360/mmseg/datasets/kitti360_pair.py```. Before running training the 2D data was converted from label ID to train ID with the script ```DeepViewAgg-WildScenes/dataset/2d/convert_train_label.py ```. Training is done with the mmsegmentation's framework file ```mmsegmentation_kitti-360/tools/train.py```:
```shell
cd mmsegmentation_kitti-360
python tools/train.py configs/kitti360/deeplabv3plus_r18-d8_512x1024_20k_k360.py --gpu-ids 0
```

All checkpoints and logs will be saved in the work_dir at ```mmsegmentation_kitti-360/work_dirs/deeplabv3plus_r18-d8_k360_40k_normal```. If you want to train with OHEM or class weights, add them into the decode head like so (according to [MMSegmentation training tricks](https://mmsegmentation.readthedocs.io/en/0.x/tutorials/training_tricks.html#online-hard-example-mining-ohem)):
```shell
decode_head=dict(                                # DeepLabV3+ head
        type='DepthwiseSeparableASPPHead',
        ...
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000))),
```
Vizualising the losses during the training is possible with the script ```/mmsegmentation_kitti-360/work_dirs/deeplabv3plus_r101-d8_k360_40k_ohem/training_loss_plot.py```. You can move this file to a work_dir of your choice and vizualise the loss AFTER training is completed. Some logs are provided for references.

### 2D-3D model
The model was used pretrained with the checkpoint provided in the DeepViewAgg repo. The name of the checkpoint is ```Res16UNet34-PointPyramid-early-cityscapes-interpolate```. The checkpoint was saved in the folder ```DeepViewAgg/checkpoints```

## Evaluation - KITTI-360

### 3D-only evaluation
Command used to obtain the results listed in the thesis:
```shell
cd DeepViewAgg
python -W ignore eval.py model_name=Res16UNet34 checkpoint_dir=path/to/checkpoint/from/training voting_runs=1 tracker_options.full_res=True tracker_options.make_submission=False precompute_multi_scale=False num_workers=8 batch_size=6 cuda=0 weight_name=Res16UNet34.pt +data.eval_sample_res=1 +data.dataroot=/home/jovyan/DeepViewAgg/dataset +pretty_print=False
```

### 2D-only evaluation
In order to run the validation with the official kitti360scripts and vizualize the results a notebook was made - ```direct_inference_kitti.ipynb```. Please refer to the notebook for more details. The notebook requires the github repo kitti360scripts:
```shell
git clone https://github.com/autonomousvision/kitti360Scripts.git
# After the prediciton images are saved from the notebook, the performance can be assessed in the terminal with:
export KITTI360_DATASET=/path/to/dataset/root
export KITTI360_RESULTS=/path/to/predictions
python /home/jovyan/DeepViewAgg/dataset/2d/kitti360Scripts/kitti360scripts/evaluation/semantic_2d/evalPixelLevelSemanticLabeling.py
```

### 2D-3D evaluation
Command used to obtain the results listed in the thesis:
```shell
python -W ignore eval.py \model_name=Res16UNet34-PointPyramid-early-cityscapes-interpolate \checkpoint_dir=/home/jovyan/DeepViewAgg/checkpoints/  \voting_runs=1 \tracker_options.full_res=True \tracker_options.make_submission=False \precompute_multi_scale=False \num_workers=8 \batch_size=6 \cuda=0  \weight_name=latest \+data.eval_sample_res=1 \+data.dataroot=/home/jovyan/DeepViewAgg/dataset \+pretty_print=False
```

## Instalation - WildScenes
This part of the thesis was done on the HPC cluster, available at the University, so in some way it differs from the one for the Jupyter server. For this to be properly configured you need to be at a node with a GPU. For the thesis, the GPU A40 was tested and used. For loading the conda and cuda modules, please refer to the [HPC Wiki](https://hpc.wiki.utwente.nl/eemcs-hpc:software).

### Adding proxies
In order for proper access to online packages the proxies should be configured as per the documentation
```shell
export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=http://proxy.utwente.nl:3128
export https_proxy=http://proxy.utwente.nl:3128
```

### Create and activate conda environemnt<br>
```shell
conda create -n deepviewagg python=3.8
conda activate deepviewagg
```

### Using GCC-9 for CUDA 11
```shell
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="8.0"
```

### Install Python dependencies<br>
```shell
pip install \
    omegaconf \
    wandb \
    tensorboard \
    plyfile \
    hydra-core==1.1.0 \
    pytorch-metric-learning \
    matplotlib \
    seaborn \
    pykeops==1.4.2 \
    imageio \
    opencv-python \
    pypng \
    git+http://github.com/CSAILVision/semantic-segmentation-pytorch.git@master \
    h5py \
    faiss-gpu==1.6.5
```

### Install PyTorch 1.7.1 and CUDA 11.0<br>
```shell
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=11.0 llvm-openm -c pytorch -c conda-forge
# Verify installation, should result into "1.7.1+cu11.0
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Install PyTorch Geometric (compatible with 1.7.1 + cu110)<br>
```shell
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install scipy
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.6.3
```

### Install additional tools
```shell
pip install torchnet gdown
pip install torch-points-kernels==0.6.10 --no-cache-dir
```

### Cd into the folder
```shell
cd DeepViewAgg-Wildscenes
```

### Clone Minkowski Engine
```shell
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd Minkowski
```

### Install sparsehash
Since for installation of Minkowski we need root rights, a work around is to install sparsehash and build it. However, for the cluster this step has additional commands that need to be run.
```shell
conda install -c conda-forge openblas
git clone https://github.com/sparsehash/sparsehash.git
cd sparsehash
export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
./configure --prefix=$HOME./local
make -j$(nproc)
make install
cd ..
conda install -y -c conda-forge libxcrypt
```

### Compile Minkowski engine
```shell
pip install ninja
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas --force_cuda
```
Test installation:
```shell
# Should print 0.5.4
python -c "import MinkowskiEngine as ME; print(ME.__version__)"
```

### (Optional) JupyterLab + Plotly
```shell
pip install plotly==5.4.0 "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.1
```

### Extra utilities
```shell
conda install plotly
conda install conda-forge::googledrivedownloader
pip install open3d
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.1.0
```

### If you need to make a notebook and use the conda environment you also need a jupyter kernel
```shell
python -m ipykernel install --user --name=deepviewagg --display-name "Python (deepviewagg)"
```

## Installation for mmsegmentation for 2D-only model

### Create and activate conda environemnt<br>
```shell
conda create -n wildscenes python=3.8 -y
conda activate wildscenes
```

### Install PyTorch 1.12 and CUDA 11.3<br>
```shell
conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

### Install MMCV and mmsegmentation
```shell
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
```

### Clone and install mmsegmentation from github
```shell
cd mmsegmentation
pip install -e .
```

### Install additional tools
```shell
pip install tqdm pandas scipy numpy pillow matplotlib imageio ipykernel
```

### Fixing issues with pretty print
There is an error with pretty print, so a file needs fixing.
```shell
nano /path/to/conda/envs/wildscenes/python3.8/site-packages/mmcv/utils/config.py
```
Go into the function "format_dict" and remove the flag ```verify=True``` on the line ```(text, _ = FormatCode(text, style_config=yapf_style, verify=True)```

## Dataset setup
Please follow the steps to download the dataset defined in the [WildScenes repo](https://github.com/csiro-robotics/WildScenes/tree/main). You would need to contact HPC administration about storing this on the cluster. Do not forget to run the script ```setup_data.py``` to correctly define the pickles for the data.

## Training - WildScenes

### 3D-only training
The model configuration is defined in ```DeepViewAgg-Wildscenes/conf/models/segmentation/multimodal/wildscenes_custom.yaml```. In order to train it, we need to be in the ```deepviewagg``` conda environment, and on a GPU node (or submit an sbatch job) and run the following command:
```shell
conda activate deepviewagg
module load nvidia/cuda-11.0
module load miniconda3/3.8
cd DeepViewAgg-WildScenes
# For whatever reason pykeops needs to be reinstalled before initiating training/evaluation on different nodes, so just to be sure run this
export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=http://proxy.utwente.nl:3128
export https_proxy=http://proxy.utwente.nl:3128

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="8.0"

export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

pip uninstall pykeops -y
pip install pykeops

python train.py \
    models=segmentation/multimodal/wildscenes_custom \
    model_name=WildScenes_3D_Only_RGB \
    data=segmentation/wildscenes_3d_only \
    task=segmentation \
    training=default \
    eval_frequency=5 \
    data.sample_per_epoch=5000 \
    data.dataroot=/path/to/root/of/WildScenes/dataset \
    data.train_is_trainval=False \
    data.mini=False \
    training.cuda=0 \
    training.batch_size=2 \
    training.epochs=60 \
    training.num_workers=0 \
    training.optim.base_lr=0.01 \
    training.wandb.log=False \
    training.wandb.name="WildScenes_3D_Baseline_Fast" \
    +training.dataloader.multiprocessing_context=spawn
```
Training should take about 90 hours.

### 2D-only training
Traning can be ran with the following command:
```shell
conda activate wildscenes
module load nvidia/cuda-11.0
module load miniconda3/3.8
cd DeepViewAgg-WildScenes/mmsegmentation
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x512_80k_wildscenes_aligned.py --gpu-ids 0
```

### 2D-3D model
After both models train and produce a checkpoint, the multimodal can be trained using ```DeepViewAgg-Wildscenes/run_fusion_training_pretrained.sh```, where path to 3D checkpoint and path to 2D checkpoint should be defined, as well as the data root. Before launching training, look into the wildscenes-advanced.py file, which requires changes in some paths (marked with TODO).

## Evaluation - WildScenes
### 3D-only evaluation
```shell
conda activate deepviewagg
module load nvidia/cuda-11.0
module load miniconda3/3.8
cd DeepViewAgg-WildScenes
# For whatever reason pykeops needs to be reinstalled before initiating training/evaluation on different nodes, so just to be sure run this
export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=http://proxy.utwente.nl:3128
export https_proxy=http://proxy.utwente.nl:3128

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="8.0"

export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

pip uninstall pykeops -y
pip install pykeops

python eval.py model_name=WildScenes_3D_Only_RGB checkpoint_dir=/path/to/checkpoint/folder num_workers=8 batch_size=6 cuda=0 weight_name=WildScenes_3D_Only_RGB.pt voting_runs=1 tracker_options.full_res=True +data.dataroot=/path/to/data/root +selection_stage=test +pretty_print=False +data.eval_sample_res=1 +multiprocessing_context=spawn
```

### 2D-only evaluation
Evaluation can be done with the following command:
```shell
conda activate wildscenes
module load nvidia/cuda-11.0
module load miniconda3/3.8
python tools/test.py \
    configs/deeplabv3/deeplabv3_r18-d8_512x512_80k_wildscenes_aligned.py \
    work_dirs/deeplabv3_r18-d8_512x512_80k_wildscenes_aligned/latest.pth \
    --eval mIoU
```

### 2D-3D evaluation
```shell
conda activate deepviewagg
module load nvidia/cuda-11.0
module load miniconda3/3.8
cd DeepViewAgg-WildScenes
# For whatever reason pykeops needs to be reinstalled before initiating training/evaluation on different nodes, so just to be sure run this
export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=http://proxy.utwente.nl:3128
export https_proxy=http://proxy.utwente.nl:3128

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=$CXX
export TORCH_CUDA_ARCH_LIST="8.0"

export CPATH=$HOME/.local/include:$CPATH
export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

pip uninstall pykeops -y
pip install pykeops

python eval.py model_name=WildScenes_LateLogitFusion_RGB checkpoint_dir=/path/to/checkpoint/ voting_runs=1 tracker_options.full_res=True num_workers=2 batch_size=4 cuda=0 weight_name=WildScenes_LateLogitFusion_RGB.pt +data.dataroot=/path/to/dataroot/ +data.eval_sample_res=1 +selection_stage=test +pretty_print=False +multiprocessing_context=spawn
```

## Acknowledgments
We would like to thank the authors of DeepViewAgg, Wildscenes, and mmsegmentation for making their code publicly available.

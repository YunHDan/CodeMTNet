<<<<<<< HEAD
# CodeMTNet
:sparkles:Official Complement of CodeMTNet.

<img width="1264" alt="model1" src="https://github.com/user-attachments/assets/5f699531-0396-41ba-86e2-1176599f22cc">

:new_moon:Code will comming in future.
=======
:sparkles:Official Complement of CodeMTNet.

<img width="1264" alt="Overview" src="Figs/Overview.png">

First, you need to execute the following command in terminal:

```bash
conda create -n your_env_name python=3.9    
conda activate your_env_name
conda install cudatoolkit==11.7 -c nvidia
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc
conda install packaging
wget https://bgithub.xyz/Dao-AILab/causal-conv1d/releases/download/v1.0.0/causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install ./causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
wget https://bgithub.xyz/state-spaces/mamba/releases/download/v1.0.1/mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install ./mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install opencv-python
pip install lpips
pip install numpy==1.23.5 --force-reinstall
```

Then you can use this command to test a photo:
```bash
cd CodeMTNet
bash run.sh
```
>>>>>>> master

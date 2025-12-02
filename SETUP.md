# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n vat python=3.10 -y
conda activate vat

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio


git clone vat
cd vat
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

if encounter error while using headless server for libero evaluation, follow instructions below:
Sudo apt-get install libegl-dev
export MUJOCO_GL=egl
create json file following https://github.com/google-deepmind/mujoco/issues/572#issuecomment-2419965230
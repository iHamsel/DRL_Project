Training can be started with: python main.py

In main.py you can select which game should be trained.

#####################################################
Lists dependencies for running the script

NumPy, Matplotlib

PyTorch:
conda install pytorch torchvision torchaudio cpuonly -c pytorch		(Pytorch, here: only cpu-usage, if possible include usage of CUDA)

Gym:
conda install -c conda-forge gym
pip install gym[atari]
conda install -c conda-forge atari_py
pip install gym[accept-rom-license]

OpenCV:
pip install opencv-python
#####################################################
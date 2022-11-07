# create virtual python environment

python -m venv venv
./activate_venv.ps1
venv\Scripts\python.exe -m pip install --upgrade pip

# install packages


pip3 install numpy
pip3 install matplotlib
pip3 install ipykernel
pip3 install coloredlogs
pip3 install tqdm

# install pytorch with cuda 11.6 support (https://pytorch.org/get-started/locally/)
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# install pytorch with cpu support only
# pip3 install torch torchvision torchaudio


# create activation file for venv in powershell (venv\Scripts\Activate.ps1)

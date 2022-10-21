# create virtual python environment

python -m venv venv
"venv\Scripts\Activate.ps1" | Out-File -FilePath "activate_venv.ps1"
./activate_venv.ps1
venv\Scripts\python.exe -m pip install --upgrade pip

# install packages


pip3 install numpy
pip3 install matplotlib


# create activation file for venv in powershell (venv\Scripts\Activate.ps1)

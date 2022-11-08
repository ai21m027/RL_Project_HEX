# RL_Project_HEX
Creating a Reinforcement Learning approach to play the game HEX using AlphaZero
https://github.com/aebrahimian/alpha-zero-hex/tree/master/hex


# Python virtual environments
The virtual environment saves all packages and python related changes inside a local folder ``'venv'`` directory which are only available in the scope where this venv is activated.

## Setup python venv
To setup a python virtual environment run the command ``"install_venv.ps1"``

If any additional packages are required add them to this file and recreate the venv.

## Recreating the venv
Just delete the venv directory and re-run ``install_venv.ps1``

## Using venv
To activate the venv open a powershell window in the repository and run ``"activate_venv.ps1"``.
In your IDE select the venv as target framework. (or run scripts directly from the activated console)

# Training

``./venv/Scripts/python.exe d:/Development/mai/RL_Project_HEX/main_train.py``

* rename and move trained models (latest checkpoint_<num>.pth.tar file) 
```
from ./pretrained_models/hex/pytorch/temp 
to ./pretrained_models/hex/pytorch
```

# Testing

``./venv/Scripts/python.exe d:/Development/mai/RL_Project_HEX/main_test.py``

# TODO
* Implement other Network than the CNN network provided from the alpha-star-general repo
* Training on Cluster for size 7x7
* Implement wrapper to play the hex_engine version of the hex game
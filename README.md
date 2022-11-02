# RL_Project_HEX
Creating a Reinforcement Learning approach to play the game HEX


# Python virtual environments
The virtual environment saves all packages and python related changes inside a local folder 'venv' directory which are only available in the scope where this venv is activated.

## Setup python venv
To setup a python virtual environment run the command "install_venv.ps1"

If any additional packages are required add them to this file and recreate the venv.

## Recreating the venv
Just delete the venv directory and re-run install_venv.ps1

## Using venv
To activate the venv open a powershell window in the repository and run "activate_venv.ps1".

(recommended): Run Visual Studio Code from this directory by entering "code ."

In your IDE select the venv as target framework. (or run scripts directly from the activated console)


# Training

Use the notebook hex_engine_training.ipynb to implement the deep Q learning algorithm there and to keep the original hex_engine.py file clean
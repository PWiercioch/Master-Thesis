@echo off

:: Script for starting tensorboard

echo activating environment

CALL conda activate master_thesis

echo reading variables

call variables.bat

echo starting tensorboard

tensorboard --logdir %model_path%/%model_type%

pause
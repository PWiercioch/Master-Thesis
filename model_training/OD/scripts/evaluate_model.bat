@echo off

:: Script for launching evaluation job on local PC

echo activating env

CALL conda activate master_thesis

echo reading variables

call variables.bat

echo setting job to run on CPU

set CUDA_VISIBLE_DEVICES=-1

echo model evaluation

python model_main_tf2.py --pipeline_config_path=%model_path%/%model_type%/%config_file% --model_dir=%model_path%/%model_type% --checkpoint_dir=%model_path%/%model_type%

pause
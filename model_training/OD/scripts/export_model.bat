@echo off

:: Script for exporting trained model

echo activating env

CALL conda activate master_thesis

echo reading variables

call variables.bat

echo setting up GPU

set CUDA_VISIBLE_DEVICES=0

echo exporting model

python exporter_main_v2.py --trained_checkpoint_dir=%model_path%/%model_type% --pipeline_config_path=%model_path%/%model_type%/%config_file% --output_directory %model_path%/%export_path%

echo model exported

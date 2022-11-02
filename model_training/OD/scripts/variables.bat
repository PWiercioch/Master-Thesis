@echo off

:: Script containing variables for the rest of batch scripts
:: Should be updated for every model to train

:: Name of the model supfolder
set model_path=efficientdet_d0
:: Name of model folder
set model_type=v1
:: Name of the config file
set config_file=ssd_efficientdet_d0_512x512_coco17_tpu-8_eval.config
:: Name of the folder to export model to
set export_path=frozen_v1
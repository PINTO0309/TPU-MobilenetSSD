#!/bin/bash

declare -A ckpt_link_map
declare -A ckpt_name_map
declare -A config_filename_map

#ckpt_link_map["mobilenet_v1_ssd"]="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz"
ckpt_link_map["mobilenet_v1_ssd"]="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz"
#ckpt_link_map["mobilenet_v2_ssd"]="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz"
ckpt_link_map["mobilenet_v2_ssd"]="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
ckpt_link_map["mobilenet_v2_ssdlite"]="http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz"

#ckpt_name_map["mobilenet_v1_ssd"]="ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18"
ckpt_name_map["mobilenet_v1_ssd"]="ssd_mobilenet_v1_coco_2018_01_28"
#ckpt_name_map["mobilenet_v2_ssd"]="ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03"
ckpt_name_map["mobilenet_v2_ssd"]="ssd_mobilenet_v2_coco_2018_03_29"
ckpt_name_map["mobilenet_v2_ssdlite"]="ssdlite_mobilenet_v2_coco_2018_05_09"

config_filename_map["mobilenet_v1_ssd-true"]="pipeline_mobilenet_v1_ssd_retrain_whole_model.config"
config_filename_map["mobilenet_v1_ssd-false"]="pipeline_mobilenet_v1_ssd_retrain_last_few_layers.config"
config_filename_map["mobilenet_v2_ssd-true"]="pipeline_mobilenet_v2_ssd_retrain_whole_model.config"
config_filename_map["mobilenet_v2_ssd-false"]="pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config"
config_filename_map["mobilenet_v2_ssdlite-true"]="pipeline_mobilenet_v2_ssdlite_retrain_whole_model.config"
config_filename_map["mobilenet_v2_ssdlite-false"]="pipeline_mobilenet_v2_ssdlite_retrain_last_few_layers.config"

INPUT_TENSORS='normalized_input_image_tensor'
OUTPUT_TENSORS='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3'

OBJ_DET_DIR="$PWD"
LEARN_DIR="${OBJ_DET_DIR}/learn"
DATASET_DIR="${LEARN_DIR}/data"
CKPT_DIR="${LEARN_DIR}/ckpt"
TRAIN_DIR="${LEARN_DIR}/train"
OUTPUT_DIR="${LEARN_DIR}/models"

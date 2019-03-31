#!/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Downloads checkpoint and dataset needed for the tutorial.

  --network_type      Can be one of [mobilenet_v1_ssd, mobilenet_v2_ssd, mobilenet_v2_ssdlite],
                      mobilenet_v1_ssd by default.
  --train_whole_model Whether or not to train all layers of the model. false
                      by default, in which only the last few layers are trained.
  --help              Display this help.
END_OF_USAGE
}

network_type="mobilenet_v1_ssd"
train_whole_model="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
      shift 2 ;;
    --train_whole_model)
      train_whole_model=$2
      shift 2;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

export PYTHONPATH=`pwd`:`pwd`/slim:$PYTHONPATH
#export PATH=/usr/local/cuda-9.0/bin:${PATH}
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH}
#ldconfig

cp pipeline_mobilenet_v1_ssd_retrain_whole_model.config configs
cp pipeline_mobilenet_v1_ssd_retrain_last_few_layers.config configs
cp pipeline_mobilenet_v2_ssd_retrain_whole_model.config configs
cp pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config configs
cp pipeline_mobilenet_v2_ssdlite_retrain_last_few_layers.config configs

source "$PWD/constants.sh"

echo "PREPARING checkpoint..."
mkdir -p "${LEARN_DIR}/ckpt"

ckpt_link="${ckpt_link_map[${network_type}]}"
ckpt_name="${ckpt_name_map[${network_type}]}"
cd "${LEARN_DIR}"
wget -O "${ckpt_name}.tar.gz" "$ckpt_link"
tar zxvf "${ckpt_name}.tar.gz"
rm "${ckpt_name}.tar.gz"
rm -rf "${CKPT_DIR}/${ckpt_name}"
rm -rf "${CKPT_DIR}/saved_model"
mv -f ${ckpt_name}/* "${CKPT_DIR}"

echo "CHOSING config file..."
config_filename="${config_filename_map[${network_type}-${train_whole_model}]}"
cd "${OBJ_DET_DIR}"
cp "configs/${config_filename}" "${CKPT_DIR}/pipeline.config"

echo "REPLACING variables in config file..."
sed -i "s%CKPT_DIR_TO_CONFIGURE%${CKPT_DIR}%g" "${CKPT_DIR}/pipeline.config"
sed -i "s%DATASET_DIR_TO_CONFIGURE%${DATASET_DIR}%g" "${CKPT_DIR}/pipeline.config"

echo "PREPARING dataset"
rm -rf "${DATASET_DIR}"
mkdir "${DATASET_DIR}"
cd "${DATASET_DIR}"

# VOCtrainval_11-May-2012.tar <--- 1.86GB
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1rATNHizJdVHnaJtt-hW9MOgjxoaajzdh" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1rATNHizJdVHnaJtt-hW9MOgjxoaajzdh" -o VOCtrainval_11-May-2012.tar

# VOCtrainval_06-Nov-2007.tar <--- 460MB
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1c8laJUn-aaWEhE5NlDwIdNv5ZdogUAcD" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1c8laJUn-aaWEhE5NlDwIdNv5ZdogUAcD" -o VOCtrainval_06-Nov-2007.tar

# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar;rm VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar;rm VOCtrainval_06-Nov-2007.tar

echo "PREPARING label map..."
cd "${OBJ_DET_DIR}"
cp "object_detection/data/pascal_label_map.pbtxt" "${DATASET_DIR}"

echo "CONVERTING dataset to TF Record..."
protoc object_detection/protos/*.proto --python_out=.
python3 object_detection/dataset_tools/create_pascal_tf_record.py \
  --label_map_path="${DATASET_DIR}/pascal_label_map.pbtxt" \
  --data_dir=${DATASET_DIR}/VOCdevkit \
  --year=merged \
  --set=train \
  --output_path="${DATASET_DIR}/pascal_train.record"

python3 object_detection/dataset_tools/create_pascal_tf_record.py \
  --label_map_path="${DATASET_DIR}/pascal_label_map.pbtxt" \
  --data_dir=${DATASET_DIR}/VOCdevkit \
  --year=merged \
  --set=val \
  --output_path="${DATASET_DIR}/pascal_val.record"

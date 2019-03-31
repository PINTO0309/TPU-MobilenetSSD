#!/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Converts TensorFlow checkpoint to EdgeTPU-compatible TFLite file.

  --network_type    Can be one of [mobilenet_v1_ssd, mobilenet_v2_ssd, mobilenet_v2_ssdlite],
                    mobilenet_v1_ssd by default.
  --checkpoint_num  Checkpoint number, by default 0.
  --help            Display this help.
END_OF_USAGE
}

network_type="mobilenet_v1_ssd"
ckpt_number=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
      shift 2 ;;
    --checkpoint_num)
      ckpt_number=$2
      shift 2 ;;
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
export PATH=/usr/local/cuda-9.0/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs:${LD_LIBRARY_PATH}
ldconfig

source "$PWD/constants.sh"

rm -rf "${OUTPUT_DIR}"
mkdir "${OUTPUT_DIR}"

echo "0 aeroplane" > "${OUTPUT_DIR}/labels.txt"
echo "1 bicycle" >> "${OUTPUT_DIR}/labels.txt"
echo "2 bird" >> "${OUTPUT_DIR}/labels.txt"
echo "3 boat" >> "${OUTPUT_DIR}/labels.txt"
echo "4 bottle" >> "${OUTPUT_DIR}/labels.txt"
echo "5 bus" >> "${OUTPUT_DIR}/labels.txt"
echo "6 car" >> "${OUTPUT_DIR}/labels.txt"
echo "7 cat" >> "${OUTPUT_DIR}/labels.txt"
echo "8 chair" >> "${OUTPUT_DIR}/labels.txt"
echo "9 cow" >> "${OUTPUT_DIR}/labels.txt"
echo "10 diningtable" >> "${OUTPUT_DIR}/labels.txt"
echo "11 dog" >> "${OUTPUT_DIR}/labels.txt"
echo "12 horse" >> "${OUTPUT_DIR}/labels.txt"
echo "13 motorbike" >> "${OUTPUT_DIR}/labels.txt"
echo "14 person" >> "${OUTPUT_DIR}/labels.txt"
echo "15 pottedplant" >> "${OUTPUT_DIR}/labels.txt"
echo "16 sheep" >> "${OUTPUT_DIR}/labels.txt"
echo "17 sofa" >> "${OUTPUT_DIR}/labels.txt"
echo "18 train" >> "${OUTPUT_DIR}/labels.txt"
echo "19 tvmonitor" >> "${OUTPUT_DIR}/labels.txt"

echo "EXPORTING frozen graph from checkpoint..."
python3 object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path="${CKPT_DIR}/pipeline.config" \
  --trained_checkpoint_prefix="${TRAIN_DIR}/model.ckpt-${ckpt_number}" \
  --output_directory="${OUTPUT_DIR}" \
  --add_postprocessing_op=true

echo "CONVERTING frozen graph to TF Lite file..."
tflite_convert \
  --output_file="${OUTPUT_DIR}/output_tflite_graph.tflite" \
  --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="${INPUT_TENSORS}" \
  --output_arrays="${OUTPUT_TENSORS}" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops

echo "TFLite graph generated at ${OUTPUT_DIR}/output_tflite_graph.tflite"

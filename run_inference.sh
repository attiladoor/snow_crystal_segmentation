
REPO_ROOT=$(git rev-parse --show-toplevel)

docker  run \
    -u $(id -u):$(id -g) \
    --gpus all \
    -v $PWD:$PWD \
    cuda-tensorflow /bin/bash -c "python3 $REPO_ROOT/run_inference_step1.py \
        --model $REPO_ROOT/models/m332_05/model.onnx \
        --input_folder $REPO_ROOT/ltu23 \
        --output_folder $REPO_ROOT/scs_out/ltu23_m332_05_paper3 "

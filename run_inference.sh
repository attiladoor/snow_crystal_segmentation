
REPO_ROOT=$(git rev-parse --show-toplevel)

docker  run \
    -u $(id -u):$(id -g) \
    --gpus all \
    -v $PWD:$PWD \
    cuda-tensorflow /bin/bash -c "python3 $REPO_ROOT/run_inference_step1.py \
        --model $REPO_ROOT/models/m324/model.onnx \
        --input_folder $REPO_ROOT/ltu24 \
        --output_folder $REPO_ROOT/scs_out/ltu24_m324 "

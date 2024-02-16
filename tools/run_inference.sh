
REPO_ROOT=$(git rev-parse --show-toplevel)

docker  run \
    -u $(id -u):$(id -g) \
    --gpus all \
    -v $PWD:$PWD \
    cuda-tensorflow /bin/bash -c "python3 $REPO_ROOT/tools/run_inference.py \
        --model $REPO_ROOT/models/m20/model.onnx \
        --input_folder $REPO_ROOT/test_png \
        --output_folder $REPO_ROOT/scs_out/m20 "

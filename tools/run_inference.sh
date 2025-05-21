
REPO_ROOT=$(git rev-parse --show-toplevel)

docker  run \
    -u $(id -u):$(id -g) \
    --gpus all \
    -v $PWD:$PWD \
    cuda-tensorflow /bin/bash -c "python3 $REPO_ROOT/run_inference.py \
        --model $REPO_ROOT/models/m241/model.onnx \
        --input_folder $REPO_ROOT/20160212 \
        --output_folder $REPO_ROOT/scs_out/20160212"

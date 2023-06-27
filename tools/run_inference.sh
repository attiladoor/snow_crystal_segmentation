
REPO_ROOT=$(git rev-parse --show-toplevel)

docker  run \
    -u $(id -u):$(id -g) \
    --gpus all \
    -v $PWD:$PWD \
    cuda-tensorflow /bin/bash -c "python3 $REPO_ROOT/tools/run_inference.py \
        --model $REPO_ROOT/out/training_06_augmented/model.onnx \
        --input_folder $REPO_ROOT/data/cropped_/batch_1/cropped_original_png \
        --output_folder $REPO_ROOT/data/cropped_/batch_1/out"
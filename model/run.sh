xhost +

DATA_FOLDER="$HOME/Downloads/cropped_/"

docker  run \
        -it \
        -u $(id -u):$(id -g) \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --gpus all \
        -v $PWD:$PWD \
        -v $DATA_FOLDER:$DATA_FOLDER \
        -p 8888:8888 \
        --name trainer \
        --rm \
        cuda-tensorflow /bin/bash -c "python3 ${PWD}/model/train.py --data_folder $DATA_FOLDER --output_folder $PWD/training-out/training"

#tensorboard --logdir_spec $PWD/build/training
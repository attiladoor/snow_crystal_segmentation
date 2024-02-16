# snow_crystal_segmentation
Semantic segmentation of ice crystals

## Pre-requisites

### Data
The original tif data should be converted to png in prior. For instructions please refer to this [LINK](https://askubuntu.com/questions/60401/batch-processing-tif-images-converting-tif-to-jpeg).

### Docker 
* install docker: [LINK](https://docs.docker.com/engine/install/ubuntu/)
* finish the post install steps: [LINK](https://docs.docker.com/engine/install/linux-postinstall/)
* install nvidia docker: [LINK](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Build the tensorflow cuda docker by:
```
cd model/docker
docker build -t cuda-tensorflow .
```

## How to run it
Modify the *DATA_FOLDER* variable in the *model/run.sh* file and just run it as
```bash
./model/run.sh
./tools/run_inference.sh
``` 

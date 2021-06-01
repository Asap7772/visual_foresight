# instructions for running this the first time:
# pull the docker image from docker hub: run ` docker pull febert/recplan:latest `
docker run --gpus all -v /raid/:/raid \
                   -v /:/parent \
                   -v /nfs:/nfs \
                   --name=${DOCKER_NAME}  \
-e VMPC_EXP=/nfs/kun1/users/febert/data/vmpc_exp \
-e VMPC_DATA=/nfs/kun1/users/febert/data/vmpc_data \
-e RECPLAN_DATA_DIR=/nfs/kun1/users/febert/data/recplan_data \
-e RECPLAN_EXP_DIR=/nfs/kun1/users/febert/data/recplan_exp \
-e RAY_RESULTS=/nfs/kun1/users/febert/data/ray_results \
-e NAS_CODE=/workspace/code \
-t -d \
--shm-size 8G \
febert/vmpc_image:latest \
/bin/bash
docker exec ${DOCKER_NAME} /bin/bash


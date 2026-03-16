#!/bin/bash


new_file=/mnt/shared-storage-user/caoweihan/projects/verl/recipe/dapo/run_dapo_qwen3_moe_30b_open_thought_fsdp_wo_ds.sh

chmod +x $new_file

set -ex
REPLICAS=8
name=n$REPLICAS-q30b-open-verl-wo-grouped-wo-ds
rjob submit -e DISTRIBUTED_JOB=true \
    --image=registry.h.pjlab.org.cn/ailab-llmrazor/xtuner:pt26_20251113_7302ba5_grouped_router_topk1 \
    --host-network=true --name $name -P $REPLICAS --gpu 8 --cpu 120  --memory 1500000 --charged-group bw_gpu \
    --private-machine='group' --namespace ailab-pj \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/caoweihan:/mnt/shared-storage-user/caoweihan \
    --mount=gpfs://gpfs1/llmrazor-share:/mnt/shared-storage-user/llmrazor-share \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs1/ailab-llmai4s:/mnt/shared-storage-user/ailab-llmai4s \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    -e SDC_TEST=true \
    --negative-tags node/gpu-lg-cmc-h-h200-0987.host.h.pjlab.org.cn \
    -- bash -ecx $new_file
 
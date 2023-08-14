# Final-project
Topdown Heatmap regression based HRNet combine CBAM for Cephalometric Landmark Detection in Lateral X-ray Images

Please focus on the hrnet.py file to see how HRnet+CBAM is implemented, and the configuration file in the configuration file home to set up the tuning parameters for the training loss.

## Since this work is based on MMPose and for the CL-detection challenge. Therefore the dataset and some accompanying files need to be downloaded separately.
First install MMPose https://mmpose.readthedocs.io/en/latest/installation.html#installation
Get the dataset on the Challenge website https://cl-detection2023.grand-challenge.org/

## Data preparation
In this step, you can execute the script step2_prepare_coco_dataset.py . Dont forget change file path.

## Model training
Change the path in the configuration file

CUDA_VISIBLE_DEVICES=0 python step3_train_and_evaluation.py \
cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py \
--work-dir='/data/xushuolin/CL-Detection2023/MMPose-checkpoints'

Base on your path change the commend. And run the demo to train.

## Test and visualiation

CUDA_VISIBLE_DEVICES=0 python step4_test_and_visualize.py \
cldetection_configs/td-hm_hrnet-w32_udp-8xb64-250e-512x512_KeypointMSELoss.py \
'/data/zhangHY/CL-Detection2023/MMPose-checkpoints/best_SDR 2.0mm_epoch_40.pth' \
--show-dir='/data/xushuolin/CL-Detection2023/MMPose-visualize' 

Same step, Base on your path change the commend. And run the demo to test save results about visualiation landmark.

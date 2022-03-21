#!/bin/sh
# now=$(date +"%Y%m%d_%H%M%S")
python3 train.py --config='configs/config_scancls.yaml' --exp_name='scanobjectnn_cls_curvenet_catnocurve' --model=curvenet

# python3 train.py --config='configs/config_scancls.yaml' --exp_name='scanobjectnn_cls_curvenet_catnocurve1' --model=curvenet

# python3 train.py
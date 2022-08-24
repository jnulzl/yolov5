export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH

# python models/tf.py --weights runs/train/Big_pose_human_detect_yolov5m_0.5/exp2/weights/best.pt --cfg models/jnulzl_models/human/yolov5m_0.5.yaml --img-size 320 320 --tf-raw-resize --source datasets/Big_pose_human_detect/images/val --tfl-int8

echo python models/tf.py --weights $1 --cfg $2 --img-size 320 320 --tf-raw-resize --source $3 --tfl-int8
python models/tf.py --weights $1 --cfg $2 --img-size 320 320 --tf-raw-resize --source $3 --tfl-int8


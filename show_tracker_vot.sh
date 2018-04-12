# pip install setproctitle
DEPLOY_PROTO='nets/tracker.prototxt'
CAFFE_MODEL='nets/models/pretrained_model/tracker.caffemodel'
TEST_DATA_PATH='data/'

python -m goturn.test.show_tracker_vot \
	--p $DEPLOY_PROTO \
	--m $CAFFE_MODEL \
	--i $TEST_DATA_PATH \
	--g 0 # change to 0 if gpu exists

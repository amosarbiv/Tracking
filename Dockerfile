FROM bvlc/caffe:cpu
#FROM bvlc/caffe:gpu
MAINTAINER orrbarkat at mail dot tau dot ac dot il

RUN apt-get update && apt-get install -y python-opencv
RUN pip install opencv-contrib-python

#https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc
VOLUME /tmp/.X11-unix:/tmp/.X11-unix

#ENTRYPOINT ["python", "-m goturn.test.show_tracker_vot", "--p nets/tracker.prototxt", "--m nets/models/pretrained_model/tracker.caffemodel", "--i data/test/", "--g 0"]
#CMD 'bash -i'

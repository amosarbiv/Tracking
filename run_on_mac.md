## prequisits:
- brew install xquartz
- brew install socat

## run:
- open -a Xquartz
- socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
- VID=path/to/video
- docker run --rm -i -t -e DISPLAY=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'):0 -v $VID:$(pwd)/data/test  -v $(pwd):$(pwd) -w $(pwd) goturn:latest bash

## TODO:
1. read movies without creating groundtruth.txt
2. add logic to fix cases when no object was detected
3. add logic to fix cases when tracker is confused
- ? add second net to verify this was the same person
- ? if not the same object enlarge search window with the last object as white pixels?
- ? other options
- ? use velocity as data to the fully connected layers
4. retrain to remove non human detections

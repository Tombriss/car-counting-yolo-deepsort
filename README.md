# Car counting with Yolo v4 and DeepSort


## Presentation

This project is widely based on [the great yolov4-deepsort repo](https://github.com/theAIGuysCode/yolov4-deepsort). It aims to detect and count vehicules captured by the camera in [Porte de la Pape, Lyon](https://www.youtube.com/watch?v=yIoY-edup8w&ab_channel=PeripheriqueNord).

![Alt Text](ressources_readme/example_output.gif)

Yolo v4 is an object detection model able to identify various types of daily objects. In our case, only cars, trucks and motorbikes are kept. The detected vehicules are then tracked by an other model called DeepSort. 

The red count corresponds to the instant number of detected vehicules on the image. The green count corresponds to the total number of vehicules seen since the beginning of the video. Some vehicules detected are not taken into account in it : a cleaning based on the total amount of time a car has been seen. 

As our models tend to perform poorly when cars are to far away or to close, the detection and tracking only occures between the green arcs : an area of optimal performances. 

In ideal conditions (sunny day, good visibility, fluid traffic), the model total count accuracy is about 95%. If the same hyperparameters of the model are kept to infer on videos in disturbed conditions (night, rainy, dense traffic), accuracy can drop significantly (about 85% if these conditions are not combined).

## Running the model

To run the model on your own videos, run ```python object_tracker.py``` with main parameters :

- ```video``` : path to the video to infer on.
- ```output```: path to output video
- ```fps_factor``` : choose 1 to handle real time. if < 1, will be faster than real time, but less accurate. if > 1 will be too slow for real time but with better accuracy. For original video with 30 fps, set it higher than 5 to compute on all frames. fps of output video approximately : fps_factor * vmoy. vmoy is the estimated average speed of the whole pipeline and is around 7. Default : 10 (running on all frames).
- ```iou``` : IOU threshold to consider bounding boxes as overlapping. Default : 0.45 (optimal for default fps_factor).
- ```score```: confidence score threshold to consider a detection. Default : 0.5 (optimal for default fps_factor).
- ```nms```: non max suppression overlap. Default : 0.5 (optimal for default fps_factor)
- ```cosine```: max cosine distance in feature space to consider that two detections are refering to the same object. Default : 0.4 (optimal for default fps_factor).
- ```onlycsv```: outputs final video and csv or just outputs csv





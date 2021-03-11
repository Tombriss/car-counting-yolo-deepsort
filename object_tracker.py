import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import pandas as pd 

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_float('fps_factor', 1.0, 'if <= 1, can handle real time. if >= 1, too slow for real time. For original video with 30 fps, set it to 5 to compute on all frames. fps of output video approximately : fps_factor * vmoy. vmoy is the estimated average speed of the whole pipeline and is around 7.')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_float('nms', 0.5, 'nms max overlap') # 1
flags.DEFINE_float('cosine', 0.2, 'max cosine distance') # 0.4
flags.DEFINE_boolean('onlycsv', False, 'computer output video or just output csv')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = FLAGS.cosine
    nms_max_overlap = FLAGS.nms

    nn_budget = 50 # max number of vehicle to keep in memory 
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        

    frame_num = 0
    computed_frame_num = 0
    # while video is running

    data = []
    df_final = pd.DataFrame(data,columns =['vehicule_id', 'frame','time','xmin','ymin','xmax','ymax','type'])

    fps_factor = FLAGS.fps_factor
    fps_original_video = vid.get(cv2.CAP_PROP_FPS)
    print("original fps : ",fps_original_video)
    vmoy = 7 # average fps of the whole pipeline

    run_every = 1 + int(fps_original_video / ( vmoy * fps_factor )) # 3
    fps_subsampled_video = fps_original_video /  run_every # 10
    timestamps = [vid.get(cv2.CAP_PROP_POS_MSEC)]

    if FLAGS.output:
        out = cv2.VideoWriter(FLAGS.output, codec, fps_original_video, (width, height))

    real_fps_pipeline = fps_subsampled_video
    fps_pipeline_list = []
    start_time = time.time()
    overall_start_time = time.time()

    while True:

        return_value, frame = vid.read()

        frame_num += 1

        if frame_num % run_every != 0:
            continue

        computed_frame_num += 1

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            timestamps.append(vid.get(cv2.CAP_PROP_POS_MSEC))
        else:
            print('Video has ended')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape

        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ["car","truck","motorbike"]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []

        for i in range(num_objects):
            bbox = bboxes[i]

            xmin,ymin,xmax,ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            xcenter = (xmin+xmax)/2
            ycenter = (ymin+ymax)/2

            if ycenter < 0.125*original_h:
                deleted_indx.append(i)
                continue


            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map


        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        r1,r2 = int(0.972*original_h), int(1.25*original_h) # detection zone : between the two circles
        center_coordinates = (original_w // 2, -int(0.694*original_h) )

        if not FLAGS.onlycsv:

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            color = (0, 255, 0)
            thickness = 1

            cv2.circle(frame, center_coordinates, r1, color, thickness)
            cv2.circle(frame, center_coordinates, r2, color, thickness)

        # update tracks
        data = []

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 10:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            xmin,ymin,xmax,ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            xcenter = (xmin+xmax)/2
            ycenter = (ymin+ymax)/2

            area_bb = abs( ( (xmax-xmin) / original_h ) * ( (ymax-ymin)  / original_w ) )
            #print(track.track_id," ",area_bb)

            rad_pos_sq = (xcenter - center_coordinates[0])**2 + (ycenter - center_coordinates[1])**2

            if r1**2 > rad_pos_sq or rad_pos_sq > r2**2 or area_bb < 0.0008 or area_bb > 0.055 :
                continue


            data.append([track.track_id,frame_num,timestamps[-1],int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),class_name])
            
        # draw bbox on screen

            if not FLAGS.onlycsv:
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                print("Center : ({},{})".format(xcenter,ycenter))

                # calculate frames per second of running detections
        if frame_num > 20:
            real_fps_pipeline = 1.0 / (time.time() - start_time)
            run_every = 1 + int(fps_original_video / ( real_fps_pipeline * fps_factor )) # 3
        start_time = time.time()
        print('Frame #: ', frame_num, "--> FPS detection : %.2f" % real_fps_pipeline, " / run every : %.2f" % run_every)
        fps_pipeline_list.append(real_fps_pipeline)


        df_data = pd.DataFrame(data,columns =['vehicule_id', 'frame','time','xmin','ymin','xmax','ymax','type'])
        df_data["fps"] = real_fps_pipeline
        df_final = df_final.append(df_data)
        clean_data = df_final.groupby("vehicule_id").filter(lambda x: len(x) > real_fps_pipeline / 2)
        n_vehicules = clean_data["vehicule_id"].unique().shape[0]

        # draw number vehicules on image
        if not FLAGS.onlycsv:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(len(data)).zfill(3), (original_w - int(0.391*original_w),int(0.139*original_h)), font, 2, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(n_vehicules), (original_w - int(0.156*original_w),int(0.139*original_h)), font, 2, (0, 255, 0), 1, cv2.LINE_AA)
        

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output and not FLAGS.onlycsv:
            for i in range(run_every):
                out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    
    if fps_pipeline_list:
        print("average fps of all pipeline : ", sum(fps_pipeline_list) / len(fps_pipeline_list))
        print("perceived total fps : ", frame_num / (time.time() - overall_start_time))
        print("execution took : ", time.time() - overall_start_time)
        
    df_final.to_csv('outputs/data.csv', index=False)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

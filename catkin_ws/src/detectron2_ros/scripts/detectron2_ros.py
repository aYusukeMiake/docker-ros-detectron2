#!/usr/bin/env python3

# Import some common libraries
# import torch
# import torchvision
import time
import numpy as np
import cv2
import random

# Import some ROS libraries
import rospy
import rosparam
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Import some common detectron2 utilities
# import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Detectron2ROS():

    def __init__(self):
        self._input_img = None
        self._is_subscribed_new_img = False
        self._task = "instance_segmentation"
        self._bridge = CvBridge()

        self.setup_rosparam()

        setup_logger()
        # Setup ROS Node
        rospy.init_node("detectron2_ros", anonymous=True)
        rospy.Subscriber("/usb_cam/image_raw", Image, self.camera_callback, queue_size=1)
        self._img_pub = rospy.Publisher("/detectron2_ros/output_img", Image, queue_size=1)
        rospy.Timer(rospy.Duration(0.01), self.predict)
        # Setup detectron2
        if (self._task == "object_detection"):
            self.setup_object_detection()
        elif (self._task == "instance_segmentation"):
            self.setup_instance_segmentation()
        elif (self._task == "panoptic_segmentation"):
            self.setup_panoptic_segmentation()
        else:
            rospy.loginfo("Incorrect task settings (string: object_detection, panoptic_segmentation, instance_segmentation)")
            return
        rospy.spin()

    def setup_rosparam(self):
        if (self._task != rosparam.get_param("/detectron2_ros/task")):
            self._task = rosparam.get_param("/detectron2_ros/task")
            if (self._task == "object_detection"):
                self.setup_object_detection()
            elif (self._task == "instance_segmentation"):
                self.setup_instance_segmentation()
            elif (self._task == "panoptic_segmentation"):
                self.setup_panoptic_segmentation()
            else:
                rospy.loginfo("Incorrect task settings (string: object_detection, panoptic_segmentation, instance_segmentation)")
                return

    def camera_callback(self, msg):
        # sensor_msgs -> cv2
        self._input_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        self._is_subscribed_new_img = True

    def predict(self, msg):
        self.setup_rosparam()
        if (self._is_subscribed_new_img == False):
            return
        if (self._task == "object_detection"):
            self.object_detection()
        elif (self._task == "instance_segmentation"):
            self.instance_segmentation()
        elif (self._task == "panoptic_segmentation"):
            self.panoptic_segmentation()
        else:
            rospy.loginfo("Incorrect task settings (string: object_detection, panoptic_segmentation, instance_segmentation)")
            return
        self._is_subscribed_new_img = True

    def setup_object_detection(self):
        self._cfg = get_cfg()
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    def object_detection(self):
        # predict
        predict_start_time = time.time()
        predictor = DefaultPredictor(self._cfg)
        outputs = predictor(self._input_img)
        v = Visualizer(self._input_img[:, :, ::-1], MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        rospy.loginfo("Predict time: {0}".format(time.time() - predict_start_time))
        # Send msg
        output_img = v.get_image()[:, :, ::-1]
        output_msg = self._bridge.cv2_to_imgmsg(output_img, encoding="bgr8")
        self._img_pub.publish(output_msg)

    def setup_instance_segmentation(self):
        self._cfg = get_cfg()
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    def instance_segmentation(self):
        # predict
        predict_start_time = time.time()
        predictor = DefaultPredictor(self._cfg)
        outputs = predictor(self._input_img)
        v = Visualizer(self._input_img[:, :, ::-1], MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        rospy.loginfo("Predict time: {0}".format(time.time() - predict_start_time))
        # Send msg
        output_img = v.get_image()[:, :, ::-1]
        output_msg = self._bridge.cv2_to_imgmsg(output_img, encoding="bgr8")
        self._img_pub.publish(output_msg)

    def setup_panoptic_segmentation(self):
        self._cfg = get_cfg()
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self._cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.7
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    def panoptic_segmentation(self):
        # predict
        predict_start_time = time.time()
        predictor = DefaultPredictor(self._cfg)
        panoptic_seg, segments_info = predictor(self._input_img)["panoptic_seg"]
        v = Visualizer(self._input_img[:, :, ::-1], MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        rospy.loginfo("Predict time: {0}".format(time.time() - predict_start_time))
        # Send msg
        output_img = v.get_image()[:, :, ::-1]
        output_msg = self._bridge.cv2_to_imgmsg(output_img, encoding="bgr8")
        self._img_pub.publish(output_msg)


if __name__ == "__main__":
    node = Detectron2ROS()

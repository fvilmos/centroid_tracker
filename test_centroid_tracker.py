#!/usr/bin/env python3

import cv2
import numpy as np
import os
import argparse

from utils.cascade import Cascade
from utils.nms import NMS
from utils.centroidtracker import CentroidTracker



if __name__ == '__main__':

    # input arguments
    parser = argparse.ArgumentParser(description='Centroid Tracker tester.\n Defaults: -cam 0 -n 0 -s 1.1 -iou 0.1 -sd 40.0')
    parser.add_argument('-cam', type=int, help='Camera ID', default=0)
    parser.add_argument('-n', type=int, help='Number of neighbors for detections', default=1)
    parser.add_argument('-s', type=float, help='Scale factor', default=1.1)
    parser.add_argument('-iou', type=float, help='Intersection Over Union threshold', default=0.1)
    parser.add_argument('-sd', type=float, help='Maximum allowed separation euclidian distance between neighbors', default=40.0)
    parser.add_argument('-rid', type=bool, help='Optimize IDs, IDs are equal to the number of objects detected. Defaults to False', default=False)


    args = parser.parse_args()

    path, filename = os.path.split(os.path.realpath(__file__))

    f = path + '/data/cascade.xml'
    cap = cv2.VideoCapture(args.cam)
    
    cd = Cascade(f)
    cd.set_parameters(args.s,args.n)

    nms = NMS()

    centd = CentroidTracker(buffer_size=20)
    
    if not cap.isOpened:
        exit(0)

    
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        if cv2.waitKey(10) == 27:
            break

        # show all detections                
        det = cd.get_detections(frame)

        for val in det:
            cv2.rectangle(frame,(val[0],val[1]),(val[0]+val[2],val[1]+val[3]),[0,255,0],1)

        # compute and show filtered detections
        det = nms.non_maximum_surpression(det, THRESHOLD=args.iou)

        for val in det:
            cv2.rectangle(frame,(val[0],val[1]),(val[0]+val[2],val[1]+val[3]),[0,0,255],1)
       
        iddet = centd.Update(det,SEPARATION_DISTIANCE=args.sd,REUSE_IDS=args.rid)

        if iddet is not None:
            for val in iddet:
                cv2.circle(frame,(val[0],val[1]),5,[0,0,255],-1)
                cv2.putText(frame, str(val[3]), (val[0]-5,val[1]-15), cv2.FONT_HERSHEY_SIMPLEX , 1, [0,0,255], 4, cv2.LINE_AA) 

        cv2.imshow('Capture', frame)

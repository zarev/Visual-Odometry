#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:29:06 2017

@author: Mariyan
"""

from pinhole_camera import PinholeCamera
from pinhole_camera import VisualOdometry

import numpy as np
import cv2
################################KITTI Dataset################################
#adapted from https://github.com/uoip/monoVO-python 

##kitti setup, dataset loading
poses_dir = 'dataset/poses/00.txt' #for ground truth
img_dir = 'dataset/sequences/00/image_0/'
cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
with open(poses_dir) as f: poses = f.readlines()#poses
print "kitti loaded."

################################Visual Odometry################################

vo = VisualOdometry(cam, poses_dir)

traj = np.zeros((600,600,3), dtype=np.uint8)

frames_arr= []
import time
start = time.time()
frames = 1000
#drawing trajectories for each frame starting form the 3rd
for img_id in range(frames):
    img = cv2.imread(img_dir+str(img_id).zfill(6)+'.png', 0)
    
    vo.update(img, img_id)

    cur_t = vo.cur_t
    
    if(img_id > 2): 
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else: 
        x, y, z = 0., 0., 0.
        
    #offset so the 2 trajectories do not overlap
    x_offset, y_offset = 0, 0
    draw_x, draw_y = int(x)+(290-x_offset), int(z)+(90-y_offset)
    true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90
       
    #openCV uses BGR colour schemes as tuples, e.g (255,0,0) is blue
    #predicted trajectory in green
    cv2.circle(traj, (draw_x,draw_y), 1, (0,255,0), 1)
    #actual trajectory in red
    cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 1)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    
    #disaplying the current coordinates in the window     
    text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

    sec = time.time()
    curr_secs = sec - start
    curr_fps = img_id/curr_secs
    frames_arr.append(curr_fps)
    
    #disaplying the current frame and FPS in the window         
    frame = "Frame: " + str(img_id) + " FPS: " + str(curr_fps)
    cv2.rectangle(traj, (10, 50), (600, 60), (0,0,0), -1)    
    cv2.putText(traj, frame, (20,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    
    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# cv2.imwrite('map.png', traj)
cv2.destroyAllWindows()
cv2.waitKey(1)

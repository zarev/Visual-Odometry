#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#adapted from https://github.com/Sentdex/pygta5
#and from: https://github.com/uoip/monoVO-python
"""
Created on Fri Apr 21 09:40:51 2017

@author: Mariyan
"""
import time
import numpy as np
import cv2
from alexnet import alexnet

#kitti setup, dataset loading
poses_dir = 'dataset/poses/00.txt' #for ground truth
img_dir = 'dataset/sequences/00/image_0/'
with open(poses_dir) as f: poses = f.readlines()#poses
print "kitti loaded."

#setting up the model variables
WIDTH=80
HEIGHT=60
LR=1e-3
EPOCHS=8
MODEL_NAME='odo-{}-{}-{}-epochs-model'.format(LR,'alexnet',EPOCHS)


model=alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)


frames=4000

traj = np.zeros((600,600,3), dtype=np.uint8)
def main():
    
    start = time.time()
    #iterating through the set of images
    for img_id in range(frames):
        start_t=time.time()#for calculating fps
        #resizing the images
        img = cv2.imread(img_dir+str(img_id).zfill(6)+'.png', 0)
        res_img=cv2.resize(img,(80,60)) 
      
        print("Frame took {} seconds.".format(time.time()-start_t))
        
        x,y,z=model.predict([res_img.reshape(WIDTH,HEIGHT,1)])[0]
#        print x*10**25, y*10**25, z*10**25
        print "Predictions (x,y,z):", x,y,z
#        print int(x*10**30),int(z*10**20)

        #offset so the 2 trajectories do not overlap
        x_offset, y_offset = 0, 0
        draw_x, draw_y = int(x)+(290-x_offset), int(z)+(90-y_offset)
        
        #extracting the true coordinates from the pose
        pose=poses[img_id].strip().split()
        trueX=float(pose[3])
        trueZ=float(pose[11])
        #fitting the values to the window size
        true_x, true_y = int(trueX)+290, int(trueZ)+90
           
        #openCV uses BGR colour schemes as tuples, e.g (255,0,0) is blue
        #predicted trajectory in green
        cv2.circle(traj, (draw_x,draw_y), 1, (0,255,0), 1)
        #actual trajectory in red
        cv2.circle(traj, (true_x,true_y), 1, (0,0,255), 1)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        
        #disaplying the current coordinates in the window     
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    
        now = time.time()
        curr_secs = now - start
        curr_fps = img_id/curr_secs
        
        #disaplying the current frame and FPS in the window         
        frame = "Frame: " + str(img_id) + " FPS: " + str(curr_fps)
        cv2.rectangle(traj, (10, 50), (600, 60), (0,0,0), -1)    
        cv2.putText(traj, frame, (20,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
        if cv2.waitKey(25) & 0xFF == ord('q'):
	            cv2.destroyAllWindows()
	            break
        


main()




# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 17:04:46 2022

@author: cynth
"""

import cv2 as cv
import time
import os

def video2frames(input_loc, output_loc):
    #Function to extract frames from an input video file and save them as separate frames in an output directory
    #Args:
        #input_loc: Input video file
        #output_loc: Output directory to save the frames
    #Returns:
        #None
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    #log the time
    time_start = time.time()
    #Start capturing the feed
    cap = cv.VideoCapture(input_loc)
    #Find the number of frames
    video_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))-1
    print('Number of frames: ', video_length)
    count = 0
    print('Converting video..\n')
    #Start converting video
    while cap.isOpened():
        #Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        #write results back to the output location
        cv.imwrite(output_loc + '/%#05d.jpg' % (count+1), frame)
        count = count+1
        #if there are no more frames left
        if (count>(video_length-1)):
            #Log the time again
            time_end = time.time()
            #Release the feed
            cap.release()
            #Print stats
            print('Done extracting frames \n%d frames extrated' % count)
            print('It took %d seconds for conversion.' % (time_end-time_start))
            break
        
if __name__=='__main__':
    input_loc = "C:/Users/cynth/Desktop/Celegans/20220425/Test/20220425test1.mp4"
    output_loc = "C:/Users/cynth/Desktop/Celegans/20220425/Test/Frames/" 
    video2frames(input_loc, output_loc) 
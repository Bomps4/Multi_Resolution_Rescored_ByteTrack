# Copyright (c) Yifu Zhang under MIT License
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

import numpy as np
import cv2

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def color_malanobis(main_color,obt_main_color):
	"""
	Malanobis distance between colors of different trackers return a weight matrix 
	"""
	main_color,obt_main_color=main_color.astype(np.float32),obt_main_color.astype(np.float32)
	
	#print('sono obt',obt_main_color)
	#print('sono main color',main_color.shape)
	first_dim=main_color.shape[0]
	second_dim=obt_main_color.shape[0]

	if(first_dim==0 or second_dim==0):
		return np.empty((first_dim,second_dim))
	final=np.array( [[cv2.compareHist(i,j,cv2.HISTCMP_CORREL)for j in obt_main_color] for  i in main_color] )
	
	return final

def ECC_matrix(main_colors,tracker_colors):
	main_colors,tracker_colors=main_colors.astype(np.float32),tracker_colors.astype(np.float32)
	first_dim=main_colors.shape[0]
	second_dim=tracker_colors.shape[0]

	if(first_dim==0 or second_dim==0):
		return np.empty((first_dim,second_dim))
	
	return np.dot(main_colors,tracker_colors.T)

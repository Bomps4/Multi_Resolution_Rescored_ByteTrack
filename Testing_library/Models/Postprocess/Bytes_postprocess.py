# Copyright (c) Yifu Zhang under MIT License
# modified by Bomps4 (luca.bompani5@unibo.it)  Copyright (C) 2023-2024 University of Bologna, Italy.

import numpy as np 
import cv2
from .similaritymetrics import iou_batch
from filterpy.kalman import KalmanFilter



def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3,low_high=False):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
 
  detection_space,detection_color=np.split(detections,np.array([5]),axis=-1)


  trackers_space,trackers_color=np.split(trackers,np.array([5]),axis=-1)
 
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8),dtype=int)
  color_matrix=1
  iou_matrix = iou_batch(detection_space, trackers_space)

  # if(low_high):
  #   color_matrix=ECC_matrix(detection_color,trackers_color)#color_malanobis(detection_color,trackers_color)

  

  #iou_matrix =iou_batch(detection_space, trackers_space)
  weight_matrix=(iou_matrix*color_matrix)
  if min(weight_matrix.shape) > 0:
  # for iou a = (iou_matrix > iou_threshold).astype(np.int32)
    a = (weight_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-weight_matrix) #for iou -iou_matrix
  #linear assignment outputs index 0 detections ,index 1 trackers
  else:
    matched_indices = np.empty(shape=(0,2))
  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(weight_matrix[m[0], m[1]]<iou_threshold):# if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,clas,rescoring):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.rescoring=rescoring
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.clas=clas
    self.retainer_score=bbox[4]
    self.score=bbox[4]
    self.aggregated_score=bbox[4]/(1-bbox[4])
  def update(self,bbox,clas):
    """
    Updates the state vector with observed bbox.
    
    """
    '''
    
    
    
    if (self.clas == clas):
      self.score = 1-((1-bbox[4])*(1-self.score))
    else:
      if(self.score<bbox[4]):
        self.clas=clas
      else:
        self.score = 1-((1-self.score)/bbox[4])
        if (self.score < 0):
          self.score=0
        
    '''
    if(self.rescoring):
      if (self.clas == clas):
        self.retainer_score = 1-((1-bbox[4])*(1-self.retainer_score))
        # self.aggregated_score*=bbox[4]/(1-bbox[4])
        # geo_mean=(self.aggregated_score)**(1/(self.hit_streak+1))
        # self.score = geo_mean/(1+geo_mean)
      else:
        if(self.retainer_score<bbox[4]):              
          self.clas=clas
        else:
          self.retainer_score = 1-((1-self.retainer_score)/(1-bbox[4])) #due probabilità distinte probabilità che la classe sia la stessa
          # self.aggregated_score*=(bbox[4])/(bbox[4])
          # geo_mean=(self.aggregated_score)**(1/(self.hit_streak+1))
          # self.score = geo_mean/(1+geo_mean) #probabilità1- che la detection sia corretta
          if (self.retainer_score < 0):
            self.retainer_score=0
          if (self.score < 0):
            self.score=0
          
      if self.retainer_score>0.9999999999:
        self.retainer_score=0.9999999999
    else:
      self.clas=clas
      self.score=bbox[4]
    
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    if(self.rescoring):
      if(bbox[4]!=0):
        # self.score=self.score if(self.score>=bbox[4]) else bbox[4]
        self.score=self.score*(self.hit_streak)/(self.hit_streak+1) + bbox[4]/(self.hit_streak+1)
      else:
        # self.score=self.score if(self.score>=bbox[4]) else bbox[4]
        self.score=self.score
    
    
    
          
    
   
    

    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)



class BYTES(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3,min_score=0.5,rescoring=True):
    """
    Sets key parameters for SORT
    """
    self.min_score=min_score
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.rescoring=rescoring
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5)),classes=np.empty((0,)),main_colors=np.empty((0,32))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    output_class=[]
    output_scores=[]
    classes=np.array(classes)

    #print('sono dets',dets)
    #input()
    high_score_dets=dets[dets[:,-1]>=self.min_score]

    high_score_classes=classes[dets[:,-1]>=self.min_score]

    high_score_data=high_score_dets

    low_score_dets=dets[dets[:,-1]<self.min_score]
    low_score_classes=classes[dets[:,-1]<self.min_score]
    
    low_score_data=low_score_dets
    
    
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:5] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    #print('sono trackers',self.trackers)
    #input()
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(high_score_data,trks, self.iou_threshold)

    #print('sono matched',matched)

    left_over=np.array([trks[i] for i in unmatched_trks])
  
	
    lwc_matched,_,_=associate_detections_to_trackers(low_score_data, left_over,self.iou_threshold)
    
    
    #matched=np.concatenate((matched,lwc_matched))
    # update matched trackers with assigned detections
    n_dets_high=high_score_dets.shape[0]
    #print(n_dets_high)
    #input()
    for m in matched:
      
      self.trackers[m[1]].update(high_score_dets[m[0], :],high_score_classes[m[0]])
    
    for m in lwc_matched:
      m=(m[0],unmatched_trks[m[1]])
      #if(self.trackers[m[1]].hit_streak >= 1):
      #low_score_dets[m[0], 4] = 0
      self.trackers[m[1]].update(low_score_dets[m[0], :],self.trackers[m[1]].clas)
    
    '''
    range_dets=np.arange(n_dets_high)
    take_but_not_analyzed=[i[0] for i in matched]
    bool_mask=np.logical_not(np.is_in(taken_unmatched,range_dets))
    range_dets=range_dets[bool_mask]
    for i in range_dets:
        ret.append(high_score_dets[i, :-1])
        output_class.append(high_score_classes[i])
        output_scores.append(high_score_dets[i, -1])
    '''
    # create and initialise new trackers for unmatched detections
	
    for i in unmatched_dets:
        d=high_score_dets[i,:-1]
        trk = KalmanBoxTracker(high_score_dets[i,:],high_score_classes[i],self.rescoring)
        self.trackers.append(trk)
        if(self.frame_count > self.min_hits):
            ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            output_class.append(trk.clas)
            output_scores.append(trk.score)
   
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        
        if (trk.time_since_update <= self.max_age and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          
          output_class.append(trk.clas)
          output_scores.append(trk.score)
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age ):
          self.trackers.pop(i)
        
    
    

    if(len(ret)>0):
      return np.concatenate((np.concatenate(ret),np.expand_dims(np.array(output_class),axis=-1),np.expand_dims(np.array(output_scores),axis=-1)),axis=-1)
    return np.concatenate((np.empty((0,5)),np.empty((0,1)),np.empty((0,1))),axis=-1)
  def clear(self):
    self.trackers = []
    self.frame_count = 0


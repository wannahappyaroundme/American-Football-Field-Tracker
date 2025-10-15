"""
SORT: Simple Online and Realtime Tracking
==========================================

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.

Reference:
    Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016).
    Simple online and realtime tracking. In 2016 IEEE International Conference on Image Processing (ICIP).

This implementation uses Kalman Filtering for state estimation and the Hungarian algorithm
for data association.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    """
    Solve the linear assignment problem using the Hungarian algorithm.
    
    Args:
        cost_matrix: Cost matrix for assignment
        
    Returns:
        matches: Array of matched indices
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.column_stack((row_ind, col_ind))


def iou_batch(bboxes1, bboxes2):
    """
    Compute IoU (Intersection over Union) between two sets of bounding boxes.
    
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    
    Args:
        bboxes1: Array of shape (N, 4) with format [x1, y1, x2, y2]
        bboxes2: Array of shape (M, 4) with format [x1, y1, x2, y2]
        
    Returns:
        iou: Array of shape (N, M) containing IoU values
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    # Compute intersection coordinates
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    
    # Compute intersection area
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # Compute union area
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection
    
    # Compute IoU
    iou = intersection / union
    
    return iou


def convert_bbox_to_z(bbox):
    """
    Convert bounding box [x1, y1, x2, y2] to [x, y, s, r] representation.
    
    Where:
        x, y: Center coordinates
        s: Scale (area)
        r: Aspect ratio (width/height)
        
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        z: Measurement vector [x, y, s, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # Scale (area)
    r = w / float(h)  # Aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert [x, y, s, r] representation back to bounding box [x1, y1, x2, y2].
    
    Args:
        x: State vector [x, y, s, r, ...]
        score: Optional confidence score
        
    Returns:
        bbox: Bounding box [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    
    Uses a Kalman filter to predict the object's next position based on its motion history.
    """
    
    count = 0  # Global track ID counter
    
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
        """
        # Define constant velocity model
        # State: [x, y, s, r, vx, vy, vs] where v denotes velocity
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r = r (aspect ratio constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1]   # vs = vs
        ])
        
        # Measurement matrix (we only observe position and scale)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.
        
        # Process noise covariance (uncertainty in motion model)
        self.kf.P[4:, 4:] *= 1000.  # High uncertainty for initial velocities
        self.kf.P *= 10.
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state with first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """
        Update the state vector with observed bounding box.
        
        Args:
            bbox: Detected bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        Advance the state vector and return the predicted bounding box estimate.
        
        Returns:
            bbox: Predicted bounding box [x1, y1, x2, y2]
        """
        # If the scale (area) is going negative, set its velocity to zero
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        # Predict next state
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """
        Return the current bounding box estimate.
        
        Returns:
            bbox: Current bounding box [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)


class Sort:
    """
    SORT Tracker - Simple Online and Realtime Tracking
    
    This tracker maintains a set of active tracks and associates new detections
    to existing tracks based on IoU (Intersection over Union).
    """
    
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep alive a track without associated detections
            min_hits: Minimum number of associated detections before track is confirmed
            iou_threshold: Minimum IoU for match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets=np.empty((0, 5))):
        """
        Update tracker with new detections.
        
        Params:
            dets: Array of detections in format [[x1, y1, x2, y2, score], ...]
            
        Returns:
            tracks: Array of active tracks in format [[x1, y1, x2, y2, track_id], ...]
        
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # Remove trackers with invalid predictions
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # Associate detections to trackers using IoU
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        
        # Return active tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Return tracks that have been hit enough times and are still active
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 for 1-indexed IDs
            
            i -= 1
            
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    
    def associate_detections_to_trackers(self, detections, trackers):
        """
        Assign detections to tracked objects (both represented as bounding boxes).
        
        Uses IoU (Intersection over Union) as the distance metric and solves the
        assignment problem using the Hungarian algorithm.
        
        Args:
            detections: Array of detections [[x1, y1, x2, y2, score], ...]
            trackers: Array of tracked objects [[x1, y1, x2, y2, score], ...]
            
        Returns:
            matched: Array of matched indices [[det_idx, trk_idx], ...]
            unmatched_detections: Array of unmatched detection indices
            unmatched_trackers: Array of unmatched tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # Compute IoU matrix
        iou_matrix = iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        # Identify unmatched detections
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        # Identify unmatched trackers
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


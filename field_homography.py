"""
Field Homography Module
=======================

Detects field lines (yard lines, hash marks, sidelines) and calculates
homography transformation to project player positions onto a 2D bird's eye
view tactical map.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class FieldLineDetector:
    """
    Detects football field lines using edge detection and Hough transform.
    """
    
    def __init__(self):
        """Initialize field line detector."""
        self.last_lines = None
    
    def detect_lines(self, frame: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect horizontal and vertical field lines.
        
        Args:
            frame: Input frame
            roi_mask: Optional ROI mask to limit detection area
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        # Apply ROI mask if provided
        if roi_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
        else:
            masked_frame = frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance white lines
        # Use top-hat morphological operation to enhance bright features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        gray_enhanced = cv2.add(gray, tophat)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 1.5)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=20
        )
        
        if lines is None or len(lines) == 0:
            return [], []
        
        # Classify lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1
            angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
            
            # Length
            length = np.sqrt(dx**2 + dy**2)
            
            line_data = {
                'coords': (x1, y1, x2, y2),
                'angle': angle,
                'length': length
            }
            
            # Classify
            if angle < 15 or angle > 165:  # Horizontal
                horizontal_lines.append(line_data)
            elif 75 < angle < 105:  # Vertical
                vertical_lines.append(line_data)
        
        # Merge similar lines
        horizontal_lines = self._merge_similar_lines(horizontal_lines, is_horizontal=True)
        vertical_lines = self._merge_similar_lines(vertical_lines, is_horizontal=False)
        
        # Cache result
        self.last_lines = (horizontal_lines, vertical_lines)
        
        return horizontal_lines, vertical_lines
    
    def _merge_similar_lines(self, lines: List[Dict], is_horizontal: bool) -> List[Dict]:
        """
        Merge similar parallel lines.
        
        Args:
            lines: List of line dictionaries
            is_horizontal: Whether lines are horizontal
            
        Returns:
            Merged lines
        """
        if not lines:
            return []
        
        # Sort by position
        if is_horizontal:
            lines_sorted = sorted(lines, key=lambda l: (l['coords'][1] + l['coords'][3]) / 2)
        else:
            lines_sorted = sorted(lines, key=lambda l: (l['coords'][0] + l['coords'][2]) / 2)
        
        merged = []
        used = [False] * len(lines_sorted)
        
        for i, line1 in enumerate(lines_sorted):
            if used[i]:
                continue
            
            group = [line1]
            used[i] = True
            
            x1_1, y1_1, x2_1, y2_1 = line1['coords']
            
            # Find similar lines
            for j in range(i + 1, len(lines_sorted)):
                if used[j]:
                    continue
                
                line2 = lines_sorted[j]
                x1_2, y1_2, x2_2, y2_2 = line2['coords']
                
                # Check distance
                if is_horizontal:
                    dist = abs((y1_1 + y2_1) / 2 - (y1_2 + y2_2) / 2)
                else:
                    dist = abs((x1_1 + x2_1) / 2 - (x1_2 + x2_2) / 2)
                
                if dist < 30:  # Merge threshold
                    group.append(line2)
                    used[j] = True
                else:
                    break  # Lines are sorted, so stop
            
            # Average the group
            avg_x1 = np.mean([l['coords'][0] for l in group])
            avg_y1 = np.mean([l['coords'][1] for l in group])
            avg_x2 = np.mean([l['coords'][2] for l in group])
            avg_y2 = np.mean([l['coords'][3] for l in group])
            avg_angle = np.mean([l['angle'] for l in group])
            total_length = sum([l['length'] for l in group])
            
            merged.append({
                'coords': (avg_x1, avg_y1, avg_x2, avg_y2),
                'angle': avg_angle,
                'length': total_length,
                'num_merged': len(group)
            })
        
        return merged


class HomographyCalculator:
    """
    Calculates homography transformation from camera view to bird's eye view.
    """
    
    def __init__(self,
                 field_length: float = 120,
                 field_width: float = 53.33,
                 output_width: int = 400,
                 output_height: int = 600):
        """
        Initialize homography calculator.
        
        Args:
            field_length: Field length in yards
            field_width: Field width in yards
            output_width: Bird's eye view width in pixels
            output_height: Bird's eye view height in pixels
        """
        self.field_length = field_length
        self.field_width = field_width
        self.output_width = output_width
        self.output_height = output_height
        
        self.homography_matrix = None
        self.last_update_frame = 0
    
    def calculate_homography(self,
                            horizontal_lines: List[Dict],
                            vertical_lines: List[Dict],
                            frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Calculate homography matrix from detected field lines.
        
        Args:
            horizontal_lines: Detected horizontal lines (yard lines)
            vertical_lines: Detected vertical lines (sidelines/hash marks)
            frame_shape: Shape of frame (height, width)
            
        Returns:
            Homography matrix or None if insufficient lines
        """
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # Select most prominent lines
        h_lines = sorted(horizontal_lines, key=lambda l: l['length'], reverse=True)[:4]
        v_lines = sorted(vertical_lines, key=lambda l: l['length'], reverse=True)[:2]
        
        # Calculate intersection points
        src_points = []
        dst_points = []
        
        # Find intersections
        for i, h_line in enumerate(h_lines[:3]):
            for j, v_line in enumerate(v_lines[:2]):
                intersection = self._line_intersection(h_line['coords'], v_line['coords'])
                
                if intersection is not None:
                    src_points.append(intersection)
                    
                    # Map to destination (bird's eye view)
                    # Assume evenly spaced yard lines
                    yard_spacing = self.output_height / len(h_lines)
                    y_dst = i * yard_spacing
                    
                    # Left or right sideline
                    x_dst = j * self.output_width
                    
                    dst_points.append((x_dst, y_dst))
        
        # Need at least 4 points for homography
        if len(src_points) < 4:
            return None
        
        # Convert to numpy arrays
        src_pts = np.float32(src_points[:4])
        dst_pts = np.float32(dst_points[:4])
        
        # Calculate homography
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        self.homography_matrix = H
        
        return H
    
    def _line_intersection(self,
                          line1: Tuple[float, float, float, float],
                          line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
        """
        Calculate intersection point of two lines.
        
        Args:
            line1: First line (x1, y1, x2, y2)
            line2: Second line (x1, y1, x2, y2)
            
        Returns:
            Intersection point (x, y) or None
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def transform_point(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Transform a point from camera view to bird's eye view.
        
        Args:
            point: Point in camera view (x, y)
            
        Returns:
            Point in bird's eye view or None if no homography
        """
        if self.homography_matrix is None:
            return None
        
        # Convert to homogeneous coordinates
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        
        # Transform
        transformed = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), self.homography_matrix)
        
        x, y = transformed[0][0]
        
        # Clip to output bounds
        x = np.clip(x, 0, self.output_width)
        y = np.clip(y, 0, self.output_height)
        
        return (float(x), float(y))
    
    def transform_points(self, points: List[Tuple[float, float]]) -> List[Optional[Tuple[float, float]]]:
        """
        Transform multiple points.
        
        Args:
            points: List of points in camera view
            
        Returns:
            List of transformed points (None for failed transformations)
        """
        return [self.transform_point(pt) for pt in points]


class BirdsEyeView:
    """
    Creates and manages bird's eye view visualization.
    """
    
    def __init__(self,
                 width: int = 400,
                 height: int = 600,
                 field_length: float = 120,
                 field_width: float = 53.33):
        """
        Initialize bird's eye view.
        
        Args:
            width: View width in pixels
            height: View height in pixels
            field_length: Field length in yards
            field_width: Field width in yards
        """
        self.width = width
        self.height = height
        self.field_length = field_length
        self.field_width = field_width
        
        self.base_field = self._create_field_template()
    
    def _create_field_template(self) -> np.ndarray:
        """
        Create base field template with yard lines.
        
        Returns:
            Field image
        """
        # Create green field
        field = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        field[:, :] = (34, 139, 34)  # Green
        
        # Draw yard lines (every 10 yards)
        yard_spacing = self.height / self.field_length
        
        for yard in range(0, int(self.field_length) + 1, 10):
            y = int(yard * yard_spacing)
            cv2.line(field, (0, y), (self.width, y), (255, 255, 255), 2)
            
            # Yard number
            cv2.putText(field, str(yard), (10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw sidelines
        cv2.rectangle(field, (0, 0), (self.width - 1, self.height - 1),
                     (255, 255, 255), 3)
        
        # Draw hash marks (simplified)
        hash_x_left = int(self.width * 0.3)
        hash_x_right = int(self.width * 0.7)
        
        for yard in range(0, int(self.field_length) + 1, 1):
            y = int(yard * yard_spacing)
            cv2.line(field, (hash_x_left - 5, y), (hash_x_left + 5, y),
                    (255, 255, 255), 1)
            cv2.line(field, (hash_x_right - 5, y), (hash_x_right + 5, y),
                    (255, 255, 255), 1)
        
        return field
    
    def draw_players(self,
                    positions: List[Tuple[float, float]],
                    teams: List[str],
                    track_ids: List[int],
                    team_colors: Dict[str, Tuple[int, int, int]],
                    radius: int = 5) -> np.ndarray:
        """
        Draw players on bird's eye view.
        
        Args:
            positions: List of (x, y) positions in bird's eye view
            teams: List of team names
            track_ids: List of track IDs
            team_colors: Dictionary mapping team names to colors
            radius: Dot radius
            
        Returns:
            Field image with players
        """
        field = self.base_field.copy()
        
        for pos, team, track_id in zip(positions, teams, track_ids):
            if pos is None:
                continue
            
            x, y = int(pos[0]), int(pos[1])
            
            # Get team color
            color = team_colors.get(team, (128, 128, 128))
            
            # Draw player dot
            cv2.circle(field, (x, y), radius, color, -1)
            cv2.circle(field, (x, y), radius + 1, (0, 0, 0), 1)  # Border
            
            # Draw ID
            cv2.putText(field, str(int(track_id)), (x + radius + 2, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return field


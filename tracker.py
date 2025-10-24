"""
Football Video Analysis Tool
=============================

Offline video analysis system for American football that combines:
1. YOLOv8 object detection for accurate player separation
2. Team classification via jersey color clustering (K-Means)
3. Static homography transformation for tactical top-down view
4. Side-by-side visualization (original + tactical map)

Priority: Accuracy and detailed analysis over speed.
Output: Processed MP4 video file with complete annotations.

Author: Computer Vision Engineer
Date: October 2025
Version: 2.0 - Clean Consolidated System
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import math
from typing import Tuple, List, Optional, Dict
import tracker_config as config  # 설정 파일 import


# ============================================================================
# PART 0: STADIUM/FIELD RECOGNITION
# ============================================================================

def create_stadium_mask(frame: np.ndarray) -> np.ndarray:
    """
    Create a binary mask that isolates the playing field/stadium area.
    
    IMPROVED: 더 강력한 경기장 밖 제외 처리
    - 녹색 필드만 정확히 검출
    - 경계 침식으로 필드 가장자리 제외
    - 작은 영역 제거로 노이즈 차단
    
    Args:
        frame: Input frame
        
    Returns:
        Binary mask (255 = field/stadium, 0 = background to exclude)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for green field
    lower_green = np.array(config.FIELD_HSV_LOWER)
    upper_green = np.array(config.FIELD_HSV_UPPER)
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE), np.uint8)
    
    # Closing: Fill gaps in the field
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, 
                                  iterations=config.MORPH_ITERATIONS)
    
    # Opening: Remove noise outside the field
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel,
                                  iterations=config.MORPH_ITERATIONS)
    
    # 작은 영역 제거 - 노이즈와 경기장 밖 작은 영역 제거
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(field_mask, connectivity=8)
    
    # 가장 큰 연결된 영역만 유지 (실제 필드)
    if num_labels > 1:
        # Find largest component (excluding background which is label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        field_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    # 필드 경계에서 안쪽으로 침식 - 경기장 밖 사람 확실히 제외
    erosion_kernel = np.ones((config.FIELD_BOUNDARY_EROSION, config.FIELD_BOUNDARY_EROSION), np.uint8)
    field_mask = cv2.erode(field_mask, erosion_kernel, iterations=1)
    
    # 최소 면적 검증
    mask_area_percent = (np.sum(field_mask > 0) / field_mask.size) * 100
    if mask_area_percent < config.MIN_FIELD_AREA_PERCENT:
        print(f"  ⚠ Warning: Field mask only covers {mask_area_percent:.1f}% (< {config.MIN_FIELD_AREA_PERCENT}%)")
    
    return field_mask


def apply_roi_mask(mask: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    """
    Apply ROI (Region of Interest) mask to exclude top and bottom percentages.
    
    Uses relative percentages rather than absolute pixels for better
    adaptability to different video resolutions.
    
    Args:
        mask: Existing mask (e.g., stadium mask)
        frame_height: Height of frame in pixels
        frame_width: Width of frame in pixels
        
    Returns:
        Modified mask with ROI applied
    """
    # Calculate boundaries based on relative percentages
    top_boundary = int(frame_height * config.ROI_TOP_PERCENT)
    bottom_boundary = int(frame_height * (1.0 - config.ROI_BOTTOM_PERCENT))
    
    # Black out top and bottom regions
    mask[0:top_boundary, :] = 0
    mask[bottom_boundary:, :] = 0
    
    return mask


def create_combined_mask(frame: np.ndarray) -> np.ndarray:
    """
    Create comprehensive mask combining stadium recognition and ROI.
    
    This creates a mask that:
    1. Identifies the green playing field (stadium masking)
    2. Excludes top percentage (scoreboard, upper crowd)
    3. Excludes bottom percentage (lower crowd, ads)
    
    Only players within this final mask will be detected.
    
    Args:
        frame: Input frame
        
    Returns:
        Final combined binary mask
    """
    height, width = frame.shape[:2]
    
    if config.ENABLE_STADIUM_MASKING:
        # Start with stadium/field mask
        mask = create_stadium_mask(frame)
        
        # Apply ROI exclusions
        mask = apply_roi_mask(mask, height, width)
    else:
        # Just use ROI mask without stadium detection
        mask = np.ones((height, width), dtype=np.uint8) * 255
        mask = apply_roi_mask(mask, height, width)
    
    return mask


# ============================================================================
# PART 1: STATIC HOMOGRAPHY CALCULATION
# ============================================================================

def detect_field_lines(frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List, List]:
    """
    Detect horizontal and vertical field lines using Hough Transform.
    
    IMPROVED: 더 정확한 필드 라인 검출
    - 더 강한 라인 강조
    - 더 나은 엣지 검출
    - 더 긴 라인만 선택
    
    Args:
        frame: Input video frame
        mask: Optional binary mask to limit detection area
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    print("  Detecting field lines (improved accuracy)...")
    
    # Apply mask if provided
    if mask is not None:
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    else:
        masked_frame = frame
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    # 흰색 라인 더 강하게 강조
    # 1. Top-hat으로 밝은 라인 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 2. 원본에 top-hat 결과를 2배로 추가 (라인 더 강조)
    enhanced = cv2.add(gray, tophat)
    enhanced = cv2.add(enhanced, tophat)
    
    # 3. Contrast 향상
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=10)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    
    # Canny edge detection (더 민감하게)
    edges = cv2.Canny(blurred, 30, 120)
    
    # Hough Line Transform (더 긴 라인만, 더 높은 threshold)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=150,
                           minLineLength=150, maxLineGap=15)
    
    if lines is None:
        return [], []
    
    # Classify lines
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
        
        if angle < 15 or angle > 165:
            horizontal_lines.append((x1, y1, x2, y2))
        elif 75 < angle < 105:
            vertical_lines.append((x1, y1, x2, y2))
    
    print(f"  Found {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
    
    return horizontal_lines, vertical_lines


def find_line_intersections(h_lines: List, v_lines: List) -> List[Tuple[float, float]]:
    """
    Find intersection points between horizontal and vertical lines.
    
    Args:
        h_lines: List of horizontal lines
        v_lines: List of vertical lines
        
    Returns:
        List of intersection points (x, y)
    """
    intersections = []
    
    for h_line in h_lines[:4]:  # Use first 4 horizontal lines
        for v_line in v_lines[:2]:  # Use first 2 vertical lines
            x1h, y1h, x2h, y2h = h_line
            x1v, y1v, x2v, y2v = v_line
            
            # Calculate intersection
            denom = (x1h - x2h) * (y1v - y2v) - (y1h - y2h) * (x1v - x2v)
            if abs(denom) < 1e-6:
                continue
            
            t = ((x1h - x1v) * (y1v - y2v) - (y1h - y1v) * (x1v - x2v)) / denom
            
            x = x1h + t * (x2h - x1h)
            y = y1h + t * (y2h - y1h)
            
            intersections.append((x, y))
    
    return intersections


def calculate_homography(first_frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Calculate static homography matrix from first frame.
    
    IMPROVED: 더 정확한 호모그래피 계산
    - 더 많은 교차점 사용
    - RANSAC으로 outlier 제거
    - 역변환 검증으로 정확도 확인
    
    Args:
        first_frame: First frame of video
        mask: Optional mask to limit line detection to field area
        
    Returns:
        Homography matrix or None if calculation fails
    """
    print("\nCalculating homography (improved accuracy for tactical view)...")
    
    # Detect field lines (using mask to focus on field area)
    h_lines, v_lines = detect_field_lines(first_frame, mask)
    
    if len(h_lines) < 3 or len(v_lines) < 2:
        print(f"  Warning: Insufficient lines (h:{len(h_lines)}, v:{len(v_lines)})")
        return None
    
    # Find intersections (source points in video)
    intersections = find_line_intersections(h_lines, v_lines)
    
    if len(intersections) < 4:
        print(f"  Warning: Insufficient intersections ({len(intersections)} < 4)")
        return None
    
    print(f"  Found {len(intersections)} intersection points")
    
    # Use more intersections for better accuracy (최대 8개 사용)
    num_points = min(len(intersections), 8)
    src_points = np.float32(intersections[:num_points])
    
    # Define destination points on top-down view
    # 실제 필드 비율을 고려한 정확한 매핑
    print(f"  Mapping {num_points} points to tactical view...")
    
    # 소스 점들을 Y 좌표로 정렬 (상단부터 하단까지)
    sorted_indices = np.argsort([pt[1] for pt in src_points])
    sorted_src = src_points[sorted_indices]
    
    # 필드 비율 계산 (실제 필드는 120야드 x 53.33야드)
    field_ratio = config.FIELD_WIDTH / config.FIELD_HEIGHT  # 400/600 = 0.67
    
    # 목적지 점들을 필드 전체에 균등하게 배치
    dst_points = []
    
    if num_points >= 4:
        # 4개 이상: 필드 전체 영역에 균등 분포
        rows = max(2, int(np.sqrt(num_points)))
        cols = max(2, (num_points + rows - 1) // rows)
        
        for i in range(num_points):
            row = i // cols
            col = i % cols
            
            # 필드 내부에 균등하게 배치 (경계에서 약간 안쪽)
            margin_x = config.FIELD_WIDTH * 0.1  # 10% 마진
            margin_y = config.FIELD_HEIGHT * 0.1
            
            x = margin_x + (col / max(1, cols - 1)) * (config.FIELD_WIDTH - 2 * margin_x)
            y = margin_y + (row / max(1, rows - 1)) * (config.FIELD_HEIGHT - 2 * margin_y)
            
            dst_points.append([x, y])
    else:
        # 4개 미만: 중앙선을 따라 배치
        center_x = config.FIELD_WIDTH / 2
        spacing_y = config.FIELD_HEIGHT / (num_points + 1)
        
        for i in range(num_points):
            y = spacing_y * (i + 1)
            dst_points.append([center_x, y])
    
    dst_points = np.float32(dst_points)
    print(f"  Destination points: {len(dst_points)} points across field")
    
    # Calculate homography matrix with RANSAC (outlier 제거)
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        print("  ✗ Homography calculation failed")
        return None
    
    # 역변환 검증 - 정확도 확인
    # 변환한 점을 다시 역변환했을 때 원본과 비슷해야 함
    transformed = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H)
    H_inv = np.linalg.inv(H)
    back_transformed = cv2.perspectiveTransform(transformed, H_inv)
    
    # 오차 계산
    error = np.mean(np.abs(src_points.reshape(-1, 1, 2) - back_transformed))
    print(f"  ✓ Homography calculated with {num_points} points (reverse error: {error:.2f} pixels)")
    
    if error > 10:
        print(f"  ⚠ Warning: High reverse error - homography may be inaccurate")
    
    return H


# ============================================================================
# PART 1.5: SIMPLE OBJECT TRACKING
# ============================================================================

class RobustTracker:
    """
    강화된 추적 시스템 - 팀 정보 캐시 및 고정
    
    핵심 기능:
    1. 각 객체에 고유 ID 부여 (절대 변경 안 됨)
    2. 팀 정보 캐시 (한번 할당되면 고정)
    3. 탐지 실패시에도 추적 유지 (캐시된 정보 사용)
    """
    
    def __init__(self, max_age=60, iou_threshold=0.25, freeze_team=True, team_confidence_frames=5):
        """
        Initialize robust tracker.
        
        Args:
            max_age: 탐지 없이 유지할 최대 프레임
            iou_threshold: IoU 매칭 임계값
            freeze_team: 팀 할당 고정 여부
            team_confidence_frames: 팀 고정까지 필요한 프레임 수
        """
        self.tracks = []  # 활성 추적 목록
        self.next_id = 1  # 다음 ID (계속 증가)
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.freeze_team = freeze_team
        self.team_confidence_frames = team_confidence_frames
    
    def calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """
        추적 업데이트 - 팀 정보 캐시 및 고정
        
        Args:
            detections: List of (bbox, team_label, team_color)
            
        Returns:
            List of tracked objects with frozen team assignments
        """
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        # 기존 추적과 새 탐지 매칭
        for track in self.tracks:
            best_iou = 0
            best_det_idx = None
            
            for det_idx in unmatched_detections:
                det_bbox = detections[det_idx][0]
                iou = self.calculate_iou(track['bbox'], det_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # 매칭 성공
            if best_iou >= self.iou_threshold:
                # Bbox 업데이트
                track['bbox'] = detections[best_det_idx][0]
                
                # 팀 정보 업데이트 (고정 로직)
                new_team_label, new_team_color = detections[best_det_idx][1], detections[best_det_idx][2]
                
                if track.get('team_frozen', False):
                    # 팀이 이미 고정됨 - 절대 바뀌지 않음!
                    pass  # 기존 팀 유지
                else:
                    # 아직 고정 안 됨 - 팀 업데이트 및 신뢰도 체크
                    if new_team_label == track.get('team_label'):
                        # 같은 팀으로 계속 분류됨 - 신뢰도 증가
                        track['team_confidence'] = track.get('team_confidence', 0) + 1
                        
                        # 충분한 신뢰도 달성 - 팀 고정!
                        if track['team_confidence'] >= self.team_confidence_frames:
                            track['team_frozen'] = True
                            if config.PRINT_FREEZE_EVENTS:
                                print(f"  🔒 ID:{track['id']} 팀 고정: {track['team_label']} (신뢰도: {track['team_confidence']}프레임)")
                    else:
                        # 팀이 바뀜 - 신뢰도 리셋하고 새 팀으로
                        track['team_label'] = new_team_label
                        track['team_color'] = new_team_color
                        track['team_confidence'] = 1
                
                track['age'] = 0
                track['hits'] = track.get('hits', 0) + 1
                matched_tracks.append(track)
                unmatched_detections.remove(best_det_idx)
            else:
                # 매칭 실패 - age 증가 (캐시된 정보 유지)
                track['age'] += 1
                if track['age'] < self.max_age:
                    # 추적 유지 (팀 정보 그대로 유지!)
                    matched_tracks.append(track)
        
        # 새로운 탐지에 대한 추적 생성
        for det_idx in unmatched_detections:
            bbox, team_label, team_color = detections[det_idx]
            matched_tracks.append({
                'id': self.next_id,
                'bbox': bbox,
                'team_label': team_label,
                'team_color': team_color,
                'team_confidence': 1,
                'team_frozen': False,  # 아직 고정 안 됨
                'age': 0,
                'hits': 1
            })
            self.next_id += 1
        
        self.tracks = matched_tracks
        return self.tracks


# ============================================================================
# PART 2: TEAM CLASSIFICATION VIA COLOR CLUSTERING
# ============================================================================

def get_team_color(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract dominant SHIRT/JERSEY color from bounding box.
    
    Focuses specifically on the shirt/jersey area (torso) and uses
    improved sampling to get accurate team colors.
    
    Args:
        frame: Full frame
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Dominant HSV color or None
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure valid bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Extract SHIRT/JERSEY region (upper-middle of bbox, avoiding head and legs)
    # Focus on 25%-60% of bbox height for best jersey visibility
    bbox_height = y2 - y1
    bbox_width = x2 - x1
    
    # Shirt region (middle-upper torso)
    shirt_y1 = int(y1 + bbox_height * 0.25)  # Start at 25% (below head)
    shirt_y2 = int(y1 + bbox_height * 0.60)  # End at 60% (above legs)
    
    # Also crop horizontally to focus on center (avoid arms/background)
    shirt_x1 = int(x1 + bbox_width * 0.2)
    shirt_x2 = int(x2 - bbox_width * 0.2)
    
    if shirt_y2 <= shirt_y1 or shirt_x2 <= shirt_x1:
        return None
    
    shirt_region = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
    
    if shirt_region.size == 0:
        return None
    
    # Convert to HSV for better color separation
    shirt_hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
    
    # Reshape to list of pixels
    pixels = shirt_hsv.reshape(-1, 3).astype(np.float32)
    
    # Filter out shadows and glare more aggressively
    brightness = pixels[:, 2]  # V channel
    saturation = pixels[:, 1]  # S channel
    
    # Keep only pixels with good saturation and reasonable brightness
    # This focuses on actual colored jersey fabric, not shadows or highlights
    mask = (brightness > 50) & (brightness < 210) & (saturation > 30)
    filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < 20:
        return None
    
    # Use K-Means with k=2 to find the two most dominant colors
    # Then select the more saturated one (likely the jersey, not skin/equipment)
    n_clusters = min(2, len(filtered_pixels))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(filtered_pixels)
    
    # Select cluster with highest average saturation (jersey color)
    best_color = None
    best_saturation = 0
    
    for center in kmeans.cluster_centers_:
        if center[1] > best_saturation:  # S channel
            best_saturation = center[1]
            best_color = center
    
    return best_color if best_color is not None else kmeans.cluster_centers_[0]


class AdaptiveTeamClassifier:
    """
    Adaptive team classifier that analyzes all players in frame.
    Groups players with similar shirt colors into teams automatically.
    """
    
    def __init__(self, n_clusters=3, min_players=6):
        """Initialize adaptive classifier."""
        self.n_clusters = n_clusters
        self.min_players = min_players
        self.team_centers = None  # Cached team color centers
        self.team_labels = {0: "Team A", 1: "Team B", 2: "Referee"}
        self.team_colors_bgr = {0: config.TEAM_A_COLOR, 1: config.TEAM_B_COLOR, 2: config.REFEREE_COLOR}
    
    def classify_all_players(self, player_colors):
        """
        Classify all players by clustering their shirt colors.
        
        Args:
            player_colors: List of HSV colors from all detected players
            
        Returns:
            List of (team_label, team_color_bgr) for each player
        """
        if len(player_colors) < self.min_players:
            # Not enough players - use fixed ranges
            return [classify_team_fixed(color) for color in player_colors]
        
        # Perform K-Means clustering on all player colors
        colors_array = np.array(player_colors)
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(player_colors)), 
                       n_init=10, random_state=42)
        labels = kmeans.fit_predict(colors_array)
        
        # Cache cluster centers for consistency
        self.team_centers = kmeans.cluster_centers_
        
        # Assign team labels based on cluster
        results = []
        for label in labels:
            team_label = self.team_labels.get(label, f"Team {label}")
            team_color = self.team_colors_bgr.get(label, config.UNKNOWN_COLOR)
            results.append((team_label, team_color))
        
        return results


def classify_team_fixed(hsv_color: np.ndarray) -> Tuple[str, Tuple[int, int, int]]:
    """
    Classify team based on fixed HSV color ranges.

    Args:
        hsv_color: Dominant HSV color (H, S, V)

    Returns:
        Tuple of (team_label, visualization_color_bgr)
    """
    h, s, v = hsv_color

    # Debug: Print detected color
    if config.DEBUG_TEAM_COLORS and np.random.rand() < 0.05:  # 5% 확률로 출력 (너무 많이 안 나오게)
        print(f"  🎨 Detected HSV: H={h:.1f}, S={s:.1f}, V={v:.1f}")

    # Check Team A range
    (h_min_a, s_min_a, v_min_a), (h_max_a, s_max_a, v_max_a) = config.TEAM_A_HSV_RANGE
    if h_min_a <= h <= h_max_a and s_min_a <= s <= s_max_a and v_min_a <= v <= v_max_a:
        if config.DEBUG_TEAM_COLORS and np.random.rand() < 0.05:
            print(f"    → Team A matched!")
        return "Team A", config.TEAM_A_COLOR

    # Check Team B range
    (h_min_b, s_min_b, v_min_b), (h_max_b, s_max_b, v_max_b) = config.TEAM_B_HSV_RANGE
    if h_min_b <= h <= h_max_b and s_min_b <= s <= s_max_b and v_min_b <= v <= v_max_b:
        if config.DEBUG_TEAM_COLORS and np.random.rand() < 0.05:
            print(f"    → Team B matched!")
        return "Team B", config.TEAM_B_COLOR

    # Check Referee range
    (h_min_r, s_min_r, v_min_r), (h_max_r, s_max_r, v_max_r) = config.REFEREE_HSV_RANGE
    if h_min_r <= h <= h_max_r and s_min_r <= s <= s_max_r and v_min_r <= v <= v_max_r:
        if config.DEBUG_TEAM_COLORS and np.random.rand() < 0.05:
            print(f"    → Referee matched!")
        return "Referee", config.REFEREE_COLOR

    if config.DEBUG_TEAM_COLORS and np.random.rand() < 0.05:
        print(f"    → Unknown (no match)")
    return "Unknown", config.UNKNOWN_COLOR


# ============================================================================
# PART 3: TOP-DOWN VIEW CREATION
# ============================================================================

def create_field_template() -> np.ndarray:
    """
    Create a blank top-down field template with yard lines.
    
    Returns:
        Field image with yard lines drawn
    """
    # Create green field
    field = np.zeros((config.FIELD_HEIGHT, config.FIELD_WIDTH, 3), dtype=np.uint8)
    field[:, :] = (34, 139, 34)  # Green
    
    # Draw yard lines every 10 yards
    yards_per_pixel = config.FIELD_HEIGHT / config.FIELD_LENGTH_YARDS
    
    for yard in range(0, config.FIELD_LENGTH_YARDS + 1, 10):
        y = int(yard * yards_per_pixel)
        cv2.line(field, (0, y), (config.FIELD_WIDTH, y), (255, 255, 255), 2)
        
        # Add yard number
        if 0 < yard < config.FIELD_LENGTH_YARDS:
            cv2.putText(field, str(yard), (10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw sidelines
    cv2.rectangle(field, (0, 0), (config.FIELD_WIDTH - 1, config.FIELD_HEIGHT - 1),
                 (255, 255, 255), 3)
    
    # Add title
    cv2.putText(field, "TACTICAL VIEW", (config.FIELD_WIDTH // 2 - 60, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return field


def transform_point_to_topdown(point: Tuple[float, float], 
                               homography: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Transform point from video coordinates to top-down view.
    
    Args:
        point: Point in video coordinates (x, y)
        homography: Homography matrix
        
    Returns:
        Transformed point or None
    """
    if homography is None:
        return None
    
    # Convert to homogeneous coordinates
    pt = np.array([[point]], dtype=np.float32)
    
    # Transform
    transformed = cv2.perspectiveTransform(pt, homography)
    
    x, y = transformed[0][0]
    
    # Clip to field bounds
    x = np.clip(x, 0, config.FIELD_WIDTH - 1)
    y = np.clip(y, 0, config.FIELD_HEIGHT - 1)
    
    return (int(x), int(y))


# ============================================================================
# PART 4: MAIN PROCESSING PIPELINE
# ============================================================================

def process_video():
    """
    Main processing pipeline - analyzes entire video offline.
    """
    print("="*70)
    print("  FOOTBALL VIDEO ANALYSIS TOOL")
    print("="*70)
    
    # Step 1: Load video
    print(f"\n[1/6] Loading video: {config.INPUT_VIDEO}")
    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {config.INPUT_VIDEO}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  ✓ Video loaded: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Step 2: Load YOLO model
    print(f"\n[2/6] Loading YOLOv8 model: {config.YOLO_MODEL}")
    model = YOLO(config.YOLO_MODEL)
    print("  ✓ Model loaded successfully")
    
    # Step 3: Analyze first frame and create stadium mask
    print("\n[3/6] Analyzing first frame for stadium/field recognition...")
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame")
        return
    
    # Create initial stadium mask from first frame
    print("  Creating stadium mask to exclude non-field areas...")
    initial_mask = create_combined_mask(first_frame)
    
    # Calculate field coverage
    mask_coverage = (np.sum(initial_mask > 0) / initial_mask.size) * 100
    print(f"  ✓ Stadium mask created - field coverage: {mask_coverage:.1f}%")
    
    if mask_coverage < 15:
        print("  ⚠ Warning: Low field coverage - consider adjusting FIELD_HSV ranges")
    
    # Calculate homography using masked field
    print("\n[4/6] Calculating static homography from first frame...")
    homography_matrix = calculate_homography(first_frame, initial_mask)
    
    if homography_matrix is None:
        print("  ⚠ Homography calculation failed - using identity matrix")
        homography_matrix = np.eye(3, dtype=np.float32)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Step 5: Create field template
    print("\n[5/6] Creating top-down field template...")
    field_template = create_field_template()
    print("  ✓ Field template created")
    
    # Step 6: Setup video writer
    print(f"\n[6/6] Setting up output video: {config.OUTPUT_VIDEO}")
    
    # Output dimensions: side-by-side (original + tactical)
    output_width = width + config.FIELD_WIDTH
    output_height = max(height, config.FIELD_HEIGHT)
    
    # 고품질 비디오 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
    out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (output_width, output_height))
    
    # VideoWriter 파라미터 설정 (품질 향상)
    if config.OUTPUT_CODEC == 'avc1':  # H.264
        # 더 높은 비트레이트로 설정 가능
        print(f"  ✓ Output: {output_width}x{output_height} @ {fps:.1f} FPS (H.264 codec, high quality)")
    else:
        print(f"  ✓ Output: {output_width}x{output_height} @ {fps:.1f} FPS")
    
    # Initialize robust tracker with team freezing
    print("\n[TRACKING] Initializing robust object tracker with team caching...")
    tracker = RobustTracker(
        max_age=config.MAX_TRACKING_FRAMES,
        iou_threshold=config.TRACKING_IOU_THRESHOLD,
        freeze_team=config.FREEZE_TEAM_ASSIGNMENT,
        team_confidence_frames=config.TEAM_ASSIGNMENT_CONFIDENCE
    ) if config.ENABLE_TRACKING else None
    if tracker:
        print("  ✓ Tracker enabled - maintains IDs and FREEZES team assignments")
        print(f"  ✓ Team freeze after {config.TEAM_ASSIGNMENT_CONFIDENCE} consistent frames")
    
    # Initialize adaptive team classifier
    print("\n[TEAM CLASSIFICATION] Initializing adaptive team classifier...")
    team_classifier = AdaptiveTeamClassifier(n_clusters=config.NUM_TEAM_CLUSTERS, min_players=config.MIN_PLAYERS_FOR_CLUSTERING)
    print("  ✓ Adaptive clustering enabled - groups similar shirt colors into teams")
    
    # Create persistent tactical map (dots stay visible, no blinking)
    # FIXED: Keep field template separate so it doesn't fade!
    print("\n[TACTICAL MAP] Setting up persistent tactical display...")
    if config.PERSISTENT_DOTS:
        # Create dots layer separate from field template
        dots_layer = np.zeros((config.FIELD_HEIGHT, config.FIELD_WIDTH, 3), dtype=np.float32)
        print("  ✓ Persistent mode - dots accumulate on separate layer (field stays clear)")
    else:
        dots_layer = None
        print("  Standard mode - map refreshes each frame")
    
    # Processing
    print(f"\n{'='*70}")
    print(f"  PROCESSING {total_frames} FRAMES")
    print("="*70)
    print("  Homography: CACHED (calculated once, reused for all frames)")
    print("  Stadium masking: ENABLED" if config.ENABLE_STADIUM_MASKING else "  Stadium masking: DISABLED")
    print(f"  ROI: Excluding top {config.ROI_TOP_PERCENT*100:.0f}% and bottom {config.ROI_BOTTOM_PERCENT*100:.0f}%")
    print("  Tracking: ENABLED - maintains IDs across frames" if config.ENABLE_TRACKING else "  Tracking: DISABLED")
    print("  Tactical dots: PERSISTENT - no blinking" if config.PERSISTENT_DOTS else "  Tactical dots: REFRESH each frame")
    print(f"  Team freeze: After {config.TEAM_ASSIGNMENT_CONFIDENCE} frames - AGGRESSIVE MODE" if config.FREEZE_TEAM_ASSIGNMENT else "  Team freeze: DISABLED")
    print("  Priority: Accuracy over speed (offline analysis)")
    print()

    frame_count = 0
    tracking_stats = {
        'total_frozen': 0,
        'max_players_tracked': 0,
        'freeze_events': []
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # ====================================================================
        # STADIUM MASKING - Recognize and isolate field area
        # ====================================================================
        
        # Create stadium mask for this frame (identifies field, excludes non-field areas)
        stadium_mask = create_combined_mask(frame)
        
        # Apply mask to frame (zeros out background)
        # This removes people/objects outside the stadium before YOLO detection
        masked_frame = cv2.bitwise_and(frame, frame, mask=stadium_mask)
        
        # ====================================================================
        # YOLO DETECTION (on masked frame with background removed)
        # ====================================================================
        
        # Run YOLO on masked frame - only detects players on the field
        results = model(masked_frame, verbose=False, conf=config.YOLO_CONFIDENCE)
        
        # ====================================================================
        # COLLECT ALL DETECTIONS AND EXTRACT COLORS
        # ====================================================================
        
        detections_raw = []  # Collect bboxes first
        player_colors = []   # Collect all colors for adaptive clustering
        ball_detections = [] # Collect ball detections
        
        # First pass: collect all detections and extract jersey colors
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                
                # Process 'person' class (class 0) and 'sports ball' class (class 32)
                if class_id == 0:  # Person
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                
                # ============================================================
                # VERIFY DETECTION IS WITHIN STADIUM MASK (강화된 검증)
                # ============================================================
                
                # 경기장 밖 사람 완전히 제외하기 위한 다중 검증
                
                # 1. 발 위치 검증 (bottom center of bbox)
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                
                if foot_x < 0 or foot_x >= stadium_mask.shape[1] or \
                   foot_y < 0 or foot_y >= stadium_mask.shape[0]:
                    continue  # Out of bounds
                
                if stadium_mask[foot_y, foot_x] == 0:
                    continue  # Foot not on field
                
                # 2. Bounding box 중심점 검증
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                if center_x < 0 or center_x >= stadium_mask.shape[1] or \
                   center_y < 0 or center_y >= stadium_mask.shape[0]:
                    continue  # Center out of bounds
                
                if stadium_mask[center_y, center_x] == 0:
                    continue  # Center not on field
                
                # 3. Bounding box 영역의 50% 이상이 필드 위에 있어야 함
                bbox_mask_region = stadium_mask[y1:y2, x1:x2]
                if bbox_mask_region.size > 0:
                    field_pixels = np.sum(bbox_mask_region > 0)
                    total_pixels = bbox_mask_region.size
                    field_percentage = field_pixels / total_pixels
                    
                    if field_percentage < 0.5:  # 50% 미만이면 제외
                        continue  # Most of bbox is outside field
                
                # Extract jersey color for this player
                dominant_color_hsv = get_team_color(frame, (x1, y1, x2, y2))
                
                if dominant_color_hsv is not None:
                    detections_raw.append((x1, y1, x2, y2, conf))
                    player_colors.append(dominant_color_hsv)
                
                elif class_id == 32 and config.ENABLE_BALL_DETECTION:  # Sports ball (football)
                    # Get bounding box for ball
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf[0])
                    
                    # Check confidence threshold for ball
                    if conf < config.BALL_CONFIDENCE:
                        continue
                    
                    # Check if ball is on field (simpler check for ball)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    if (0 <= center_x < stadium_mask.shape[1] and 
                        0 <= center_y < stadium_mask.shape[0] and
                        stadium_mask[center_y, center_x] > 0):
                        
                        ball_detections.append((x1, y1, x2, y2, conf))
                        print(f"  🏈 Ball detected at ({center_x}, {center_y}) with confidence {conf:.2f}")
        
        # ====================================================================
        # ADAPTIVE TEAM CLASSIFICATION - Cluster all players together
        # ====================================================================
        
        detections_list = []  # For tracker: (bbox, team_label, team_color)
        
        if len(player_colors) > 0:
            # Use adaptive clustering to classify all players at once
            if config.TEAM_DETECTION_METHOD == 'adaptive_clustering':
                team_classifications = team_classifier.classify_all_players(player_colors)
            else:
                team_classifications = [classify_team_fixed(color) for color in player_colors]
            
            # Combine bboxes with team classifications
            for i, (x1, y1, x2, y2, conf) in enumerate(detections_raw):
                team_label, box_color = team_classifications[i]
                detections_list.append(((x1, y1, x2, y2), team_label, box_color))
        
        # ====================================================================
        # UPDATE TRACKER - Maintain objects across frames
        # ====================================================================
        
        if tracker:
            tracked_objects = tracker.update(detections_list)

            # Update tracking statistics
            frozen_count = sum(1 for obj in tracked_objects if obj.get('team_frozen', False))
            total_count = len(tracked_objects)
            tracking_stats['total_frozen'] = max(tracking_stats['total_frozen'], frozen_count)
            tracking_stats['max_players_tracked'] = max(tracking_stats['max_players_tracked'], total_count)

            # Print tracking statistics (every 30 frames)
            if config.PRINT_TRACKING_STATS and frame_count % 30 == 0:
                fresh_count = sum(1 for obj in tracked_objects if obj.get('age', 0) == 0)
                print(f"  📊 Frame {frame_count}: {total_count} players | {frozen_count} frozen (🔒) | {fresh_count} fresh detections")
        else:
            # No tracking - use detections directly
            tracked_objects = [{'id': i, 'bbox': d[0], 'team_label': d[1], 'team_color': d[2], 'age': 0}
                             for i, d in enumerate(detections_list)]

        # ====================================================================
        # VISUALIZATION - Draw tracked objects
        # ====================================================================
        
        # FIXED: Start with fresh field template (never fades)
        topdown_view = field_template.copy()
        
        # If persistent dots enabled, fade and overlay the dots layer
        if config.PERSISTENT_DOTS and dots_layer is not None:
            # Fade old dots only (not the field!)
            dots_layer = dots_layer * config.DOT_FADE_ALPHA
            
            # Add faded dots to fresh field template
            mask = (dots_layer > 0).any(axis=2).astype(np.uint8) * 255
            topdown_view = cv2.addWeighted(topdown_view, 1.0, 
                                          dots_layer.astype(np.uint8), 1.0, 0)
        
        # Annotate original frame
        annotated_frame = frame.copy()
        
        # Draw each tracked object
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            team_label = obj['team_label']
            box_color = obj['team_color']
            track_id = obj['id']
            
            # ============================================================
            # DRAW ON ORIGINAL FRAME
            # ============================================================

            # Draw bounding box with team color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)

            # Add green border indicator for frozen teams
            is_frozen = obj.get('team_frozen', False)
            if config.SHOW_FROZEN_INDICATOR and is_frozen:
                # Draw thick green outer border to indicate frozen team
                cv2.rectangle(annotated_frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 2)

            # Draw label with ID and team (add 🔒 emoji for frozen)
            lock_emoji = "🔒" if is_frozen else ""
            label = f"ID:{track_id} {team_label} {lock_emoji}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 10, y1), box_color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # ============================================================
            # TRANSFORM TO TOP-DOWN VIEW
            # ============================================================
            
            # Calculate foot position (bottom center of bbox)
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            # Transform to top-down coordinates using CACHED homography
            topdown_point = transform_point_to_topdown((foot_x, foot_y), homography_matrix)
            
            if topdown_point is not None:
                # Draw player position on tactical map AND dots layer
                cv2.circle(topdown_view, topdown_point, 6, box_color, -1)
                cv2.circle(topdown_view, topdown_point, 7, (0, 0, 0), 1)
                
                # Draw ID number
                cv2.putText(topdown_view, str(track_id), 
                           (topdown_point[0] + 8, topdown_point[1] + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Also draw on dots layer for persistence
                if config.PERSISTENT_DOTS and dots_layer is not None:
                    cv2.circle(dots_layer, topdown_point, 6, box_color, -1)
                    cv2.circle(dots_layer, topdown_point, 7, (0, 0, 0), 1)
        
        # ====================================================================
        # DRAW FOOTBALL BALL DETECTIONS
        # ====================================================================
        
        for ball_bbox in ball_detections:
            x1, y1, x2, y2, conf = ball_bbox
            
            # Draw ball on original frame
            ball_color = config.BALL_COLOR  # Ball color from config
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ball_color, 3)
            
            # Ball label
            ball_label = f"🏈 Ball ({conf:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(ball_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), 
                         (x1 + text_w + 10, y1), ball_color, -1)
            cv2.putText(annotated_frame, ball_label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Transform ball position to tactical view
            ball_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            ball_topdown = transform_point_to_topdown(ball_center, homography_matrix)
            
            if ball_topdown is not None:
                # Draw ball on tactical view
                cv2.circle(topdown_view, ball_topdown, 8, ball_color, -1)
                cv2.circle(topdown_view, ball_topdown, 9, (0, 0, 0), 2)
                cv2.putText(topdown_view, "🏈", (ball_topdown[0] - 6, ball_topdown[1] + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Also draw on dots layer for persistence
                if config.PERSISTENT_DOTS and dots_layer is not None:
                    cv2.circle(dots_layer, ball_topdown, 8, ball_color, -1)
                    cv2.circle(dots_layer, ball_topdown, 9, (0, 0, 0), 2)
        
        # ====================================================================
        # CREATE SIDE-BY-SIDE OUTPUT
        # ====================================================================
        
        # Resize tactical view to match frame height if needed
        if topdown_view.shape[0] != height:
            topdown_resized = cv2.resize(topdown_view, (config.FIELD_WIDTH, height))
        else:
            topdown_resized = topdown_view
        
        # Combine original (annotated) and tactical view side-by-side
        combined = np.hstack([annotated_frame, topdown_resized])
        
        # Ensure correct output dimensions
        if combined.shape[0] != output_height or combined.shape[1] != output_width:
            combined = cv2.resize(combined, (output_width, output_height))
        
        # Write to output video
        out.write(combined)
        
        # Optional: Show preview (comment out for faster processing)
        # cv2.imshow('Processing...', cv2.resize(combined, (1280, 720)))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n  ✓ Processing complete!")
    print(f"\n{'='*70}")
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput saved to: {config.OUTPUT_VIDEO}")
    print(f"Total frames processed: {frame_count}")

    # Print team freezing statistics
    if tracker and config.FREEZE_TEAM_ASSIGNMENT:
        print(f"\n{'='*70}")
        print("  TEAM FREEZING STATISTICS")
        print("="*70)
        print(f"Maximum players tracked simultaneously: {tracking_stats['max_players_tracked']}")
        print(f"Maximum frozen players: {tracking_stats['total_frozen']}")
        freeze_rate = (tracking_stats['total_frozen'] / tracking_stats['max_players_tracked'] * 100) if tracking_stats['max_players_tracked'] > 0 else 0
        print(f"Freeze success rate: {freeze_rate:.1f}%")
        print(f"Team freeze threshold: {config.TEAM_ASSIGNMENT_CONFIDENCE} consecutive frames")
        print(f"\n✅ Players with green borders (🔒) have permanently frozen team assignments")
        print(f"✅ These players will NEVER change teams during the video")

    print(f"\nOpen {config.OUTPUT_VIDEO} to view the analysis.")
    print("\nFeatures in output:")
    print("  • Left side: Original video with team-colored bounding boxes")
    print("  • Right side: Top-down tactical view showing player positions")
    print("  • Color coding: Team A (blue), Team B (red), Referee (yellow)")
    if config.SHOW_FROZEN_INDICATOR:
        print("  • Green border: Player team is permanently frozen 🔒")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        process_video()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

"""
CLIP-based classifier for football play analysis.
Uses OpenAI CLIP for semantic entity classification and team identification.
"""

import clip
import torch
import cv2
import numpy as np
from PIL import Image
from config import (
    CLIP_MODEL_NAME,
    CLIP_DEVICE,
    CLIP_ENTITY_PROMPTS,
    CLIP_TEAM_PROMPTS,
    CLIP_BATCH_SIZE,
    CLIP_CONFIDENCE_THRESHOLD,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    REFEREE_COLOR,
    UNKNOWN_COLOR
)


class CLIPEntityClassifier:
    """
    Classifies entities in football footage using CLIP zero-shot classification.
    Distinguishes between players, referees, and sideline personnel.
    """

    def __init__(self):
        """
        Initialize the CLIP entity classifier.
        """
        print(f"Loading CLIP model '{CLIP_MODEL_NAME}' on device '{CLIP_DEVICE}'...")
        self.device = CLIP_DEVICE
        self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)

        # Entity classification prompts
        self.entity_prompts = CLIP_ENTITY_PROMPTS

        # Category mapping
        self.categories = ['player', 'referee', 'sideline', 'other']

        # Pre-encode text prompts for efficiency
        self.text_features = self._encode_prompts()

        print(f"CLIP Entity Classifier initialized with {len(self.entity_prompts)} prompts")

    def _encode_prompts(self):
        """
        Pre-encode text prompts to improve inference speed.

        Returns:
            Normalized text feature embeddings
        """
        text_inputs = clip.tokenize(self.entity_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify_entity(self, image_crop):
        """
        Classify what type of entity is in the crop.

        Args:
            image_crop: numpy array (H, W, 3) in BGR format from OpenCV

        Returns:
            Tuple of (entity_type, confidence):
                entity_type: 'player', 'referee', 'sideline', or 'other'
                confidence: float between 0 and 1
        """
        # Check for valid crop
        if image_crop is None or image_crop.size == 0:
            return 'other', 0.0

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        try:
            image_pil = Image.fromarray(image_rgb)
        except Exception as e:
            print(f"Warning: Failed to convert crop to PIL Image: {e}")
            return 'other', 0.0

        # Preprocess image
        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        # Get best match
        probs = similarity[0].cpu().numpy()
        best_idx = probs.argmax()
        confidence = float(probs[best_idx])

        # Map to category
        entity_type = self.categories[best_idx] if best_idx < len(self.categories) else 'other'

        return entity_type, confidence

    def batch_classify_entities(self, image_crops, track_ids=None):
        """
        Efficiently classify multiple crops in a single batch.

        Args:
            image_crops: List of numpy arrays (H, W, 3) in BGR format
            track_ids: Optional list of track IDs for logging (same length as image_crops)

        Returns:
            List of tuples (entity_type, confidence) in same order as input crops
        """
        if len(image_crops) == 0:
            return []

        results = []
        processed_images = []
        valid_indices = []

        # Preprocess all images
        for idx, crop in enumerate(image_crops):
            if crop is None or crop.size == 0:
                results.append(('other', 0.0))
                continue

            try:
                image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                processed_images.append(self.preprocess(image_pil))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to preprocess crop {idx}: {e}")
                results.append(('other', 0.0))

        if len(processed_images) == 0:
            return results

        # Process in batches
        batch_size = CLIP_BATCH_SIZE
        batch_results = [None] * len(valid_indices)

        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(batch_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

            # Extract results for this batch
            for j in range(len(batch)):
                probs = similarity[j].cpu().numpy()
                best_idx = probs.argmax()
                confidence = float(probs[best_idx])
                entity_type = self.categories[best_idx] if best_idx < len(self.categories) else 'other'
                batch_results[i + j] = (entity_type, confidence)

        # Merge results back in original order
        final_results = []
        valid_idx = 0
        for idx in range(len(image_crops)):
            if idx in valid_indices:
                final_results.append(batch_results[valid_idx])
                valid_idx += 1
            else:
                final_results.append(('other', 0.0))

        return final_results


class CLIPTeamClassifier:
    """
    Classifies players into teams using CLIP zero-shot classification.
    Alternative to color-based K-means clustering.
    """

    def __init__(self):
        """
        Initialize the CLIP team classifier.
        """
        print(f"Loading CLIP model '{CLIP_MODEL_NAME}' on device '{CLIP_DEVICE}'...")
        self.device = CLIP_DEVICE
        self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)

        # Team classification prompts
        self.team_prompts = CLIP_TEAM_PROMPTS

        # Team labels and colors
        self.team_labels = ['Team A', 'Team B', 'Referee']
        self.team_colors = {
            'Team A': TEAM_A_COLOR,
            'Team B': TEAM_B_COLOR,
            'Referee': REFEREE_COLOR,
            'Unknown': UNKNOWN_COLOR
        }

        # Pre-encode text features
        self.text_features = self._encode_prompts()

        # Storage for player team assignments
        self.player_teams = {}  # track_id -> team_label

        # Team freeze mechanism (팀 고정 메커니즘)
        self.frozen_teams = {}  # track_id -> team_label (고정된 팀, 변경 불가)
        self.team_confidence_history = {}  # track_id -> [(team, conf), ...]
        self.freeze_threshold = 1  # 1프레임 즉시 고정! (처음 1번 감지 후 변경 불가)

        print(f"CLIP Team Classifier initialized with {len(self.team_prompts)} team prompts")
        print(f"Team freeze mechanism: IMMEDIATE (1-frame freeze)")

    def _encode_prompts(self):
        """
        Pre-encode text prompts for team classification.

        Returns:
            Normalized text feature embeddings
        """
        text_inputs = clip.tokenize(self.team_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def classify_team(self, crop_image):
        """
        Classify team using CLIP instead of color clustering.

        Args:
            crop_image: numpy array (H, W, 3) in BGR format

        Returns:
            Tuple of (team_label, team_color, confidence):
                team_label: 'Team A', 'Team B', 'Referee', or 'Unknown'
                team_color: BGR tuple
                confidence: float between 0 and 1
        """
        if crop_image is None or crop_image.size == 0:
            return 'Unknown', self.team_colors['Unknown'], 0.0

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        try:
            image_pil = Image.fromarray(image_rgb)
        except Exception as e:
            print(f"Warning: Failed to convert crop to PIL Image: {e}")
            return 'Unknown', self.team_colors['Unknown'], 0.0

        # Preprocess
        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        # Encode and classify
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        best_idx = probs.argmax()
        confidence = float(probs[best_idx])

        # Get team label and color
        team_label = self.team_labels[best_idx] if best_idx < len(self.team_labels) else 'Unknown'
        team_color = self.team_colors[team_label]

        return team_label, team_color, confidence

    def assign_team(self, track_id, crop_image):
        """
        Assign a team to a tracked player with freeze mechanism.
        Once a team is frozen (2 consecutive high-confidence frames), it cannot be changed.

        Args:
            track_id: The track ID of the player
            crop_image: Cropped image of the player

        Returns:
            Tuple of (team_label, team_color, confidence)
        """
        # ===== 프리즈된 팀은 절대 변경하지 않음! =====
        if track_id in self.frozen_teams:
            frozen_team = self.frozen_teams[track_id]
            frozen_color = self.team_colors[frozen_team]
            return frozen_team, frozen_color, 1.0  # 프리즈된 팀은 신뢰도 1.0

        # CLIP으로 팀 분류
        team_label, team_color, confidence = self.classify_team(crop_image)

        # 신뢰도 히스토리 기록 (최대 300 프레임 = 10초 at 30 FPS)
        if track_id not in self.team_confidence_history:
            self.team_confidence_history[track_id] = []

        self.team_confidence_history[track_id].append((team_label, confidence))

        # 히스토리 길이 제한 (메모리 관리 + 장기 추적 지원)
        max_history = 300  # 10초 히스토리 유지
        if len(self.team_confidence_history[track_id]) > max_history:
            self.team_confidence_history[track_id] = self.team_confidence_history[track_id][-max_history:]

        # 최근 N프레임 확인 (프리즈 조건 체크)
        history = self.team_confidence_history[track_id][-self.freeze_threshold:]

        # 프리즈 조건: 즉시 고정! (첫 감지 시 바로 고정)
        if len(history) >= self.freeze_threshold:
            # 모든 프레임에서 동일한 팀인지 확인
            if all(h[0] == team_label for h in history):
                # 신뢰도가 0.3 이상이면 즉시 고정 (낮은 임계값으로 빠른 고정)
                if all(h[1] >= 0.3 for h in history):
                    # 팀 프리즈!
                    self.frozen_teams[track_id] = team_label
                    self.player_teams[track_id] = team_label
                    print(f"🔒 Team FROZEN: Track #{track_id} → {team_label} (conf: {confidence:.2%})")

        # 프리즈되지 않은 경우에만 임시 저장 (confidence threshold 체크)
        if confidence >= CLIP_CONFIDENCE_THRESHOLD:
            self.player_teams[track_id] = team_label

        return team_label, team_color, confidence

    def is_frozen(self, track_id):
        """
        Check if a team assignment is frozen (cannot be changed).

        Args:
            track_id: The track ID of the player

        Returns:
            Boolean: True if team is frozen, False otherwise
        """
        return track_id in self.frozen_teams

    def get_team(self, track_id):
        """
        Get the team assignment for a track ID.

        Args:
            track_id: The track ID of the player

        Returns:
            Team name string ('Team A', 'Team B', 'Referee', 'Unknown')
        """
        return self.player_teams.get(track_id, 'Unknown')

    def get_team_color(self, team_label):
        """
        Get the BGR color for a team label.

        Args:
            team_label: Team name string

        Returns:
            BGR color tuple
        """
        return self.team_colors.get(team_label, self.team_colors['Unknown'])

    def batch_classify_teams(self, image_crops, track_ids=None):
        """
        Efficiently classify teams for multiple crops in batches.

        Args:
            image_crops: List of numpy arrays (H, W, 3) in BGR format
            track_ids: Optional list of track IDs to store assignments

        Returns:
            List of tuples (team_label, team_color, confidence)
        """
        if len(image_crops) == 0:
            return []

        results = []
        processed_images = []
        valid_indices = []

        # Preprocess all images
        for idx, crop in enumerate(image_crops):
            if crop is None or crop.size == 0:
                results.append(('Unknown', self.team_colors['Unknown'], 0.0))
                continue

            try:
                image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                processed_images.append(self.preprocess(image_pil))
                valid_indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to preprocess crop {idx}: {e}")
                results.append(('Unknown', self.team_colors['Unknown'], 0.0))

        if len(processed_images) == 0:
            return results

        # Process in batches
        batch_size = CLIP_BATCH_SIZE
        batch_results = [None] * len(valid_indices)

        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i+batch_size]
            batch_tensor = torch.stack(batch).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(batch_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

            # Extract results for this batch
            for j in range(len(batch)):
                probs = similarity[j].cpu().numpy()
                best_idx = probs.argmax()
                confidence = float(probs[best_idx])
                team_label = self.team_labels[best_idx] if best_idx < len(self.team_labels) else 'Unknown'
                team_color = self.team_colors[team_label]
                batch_results[i + j] = (team_label, team_color, confidence)

        # Merge results back in original order and store assignments
        final_results = []
        valid_idx = 0
        for idx in range(len(image_crops)):
            if idx in valid_indices:
                result = batch_results[valid_idx]
                final_results.append(result)

                # Store assignment if track_ids provided and confidence is sufficient
                if track_ids is not None and result[2] >= CLIP_CONFIDENCE_THRESHOLD:
                    self.player_teams[track_ids[idx]] = result[0]

                valid_idx += 1
            else:
                final_results.append(('Unknown', self.team_colors['Unknown'], 0.0))

        return final_results

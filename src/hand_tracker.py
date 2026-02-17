"""
Hand Tracking Module
Tracks human hand pose for robotic arm control
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional


class HandTracker:
    """Tracks hand pose and extracts control signals"""
    
    def __init__(self):
        """Initialize hand tracker"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def track_hand(self, frame: np.ndarray) -> Dict:
        """
        Track hand in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            dict: Hand tracking results with landmarks and control signals
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        result = {
            'landmarks': None,
            'hand_detected': False,
            'control_signals': {},
            'gesture': None
        }
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            result['landmarks'] = hand_landmarks
            result['hand_detected'] = True
            
            # Extract control signals
            result['control_signals'] = self._extract_control_signals(hand_landmarks, frame.shape)
            result['gesture'] = self._detect_gesture(hand_landmarks)
        
        return result
    
    def _extract_control_signals(self, landmarks, frame_shape: Tuple) -> Dict:
        """
        Extract control signals from hand landmarks
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            dict: Control signals for robotic arm
        """
        # Get key points
        wrist = landmarks.landmark[0]  # Wrist
        index_tip = landmarks.landmark[8]  # Index finger tip
        middle_tip = landmarks.landmark[12]  # Middle finger tip
        ring_tip = landmarks.landmark[16]  # Ring finger tip
        pinky_tip = landmarks.landmark[20]  # Pinky tip
        thumb_tip = landmarks.landmark[4]  # Thumb tip
        
        # Normalize coordinates (0-1)
        h, w = frame_shape[:2]
        
        # Target position (index finger tip)
        target_x = index_tip.x
        target_y = index_tip.y
        target_z = index_tip.z  # Depth estimate
        
        # Wrist position (base reference)
        base_x = wrist.x
        base_y = wrist.y
        base_z = wrist.z
        
        # Calculate relative position
        rel_x = target_x - base_x
        rel_y = target_y - base_y
        rel_z = target_z - base_z
        
        # Hand orientation (using middle finger direction)
        middle_mcp = landmarks.landmark[9]
        direction_x = middle_tip.x - middle_mcp.x
        direction_y = middle_tip.y - middle_mcp.y
        direction_z = middle_tip.z - middle_mcp.z
        
        # Calculate angles
        yaw = np.arctan2(direction_y, direction_x)  # Rotation around Z
        pitch = np.arctan2(direction_z, np.sqrt(direction_x**2 + direction_y**2))  # Rotation around Y
        
        # Gripper state (distance between thumb and index)
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        # Normalize gripper (0 = closed, 1 = open)
        gripper_state = np.clip(thumb_index_dist * 5, 0, 1)
        
        return {
            'target_position': {
                'x': float(target_x),
                'y': float(target_y),
                'z': float(target_z)
            },
            'base_position': {
                'x': float(base_x),
                'y': float(base_y),
                'z': float(base_z)
            },
            'relative_position': {
                'x': float(rel_x),
                'y': float(rel_y),
                'z': float(rel_z)
            },
            'orientation': {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': 0.0  # Would need additional landmarks
            },
            'gripper': float(gripper_state),
            'fingers': {
                'index': self._is_finger_extended(landmarks, 8),
                'middle': self._is_finger_extended(landmarks, 12),
                'ring': self._is_finger_extended(landmarks, 16),
                'pinky': self._is_finger_extended(landmarks, 20),
                'thumb': self._is_thumb_extended(landmarks)
            }
        }
    
    def _is_finger_extended(self, landmarks, tip_idx: int) -> bool:
        """Check if finger is extended"""
        # Finger tip, PIP, MCP indices
        finger_indices = {
            8: (8, 6, 5),   # Index
            12: (12, 10, 9),  # Middle
            16: (16, 14, 13),  # Ring
            20: (20, 18, 17)   # Pinky
        }
        
        if tip_idx not in finger_indices:
            return False
        
        tip, pip, mcp = finger_indices[tip_idx]
        tip_y = landmarks.landmark[tip].y
        pip_y = landmarks.landmark[pip].y
        mcp_y = landmarks.landmark[mcp].y
        
        return tip_y < pip_y < mcp_y
    
    def _is_thumb_extended(self, landmarks) -> bool:
        """Check if thumb is extended"""
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        
        # Thumb uses x coordinate
        return thumb_tip.x > thumb_ip.x > thumb_mcp.x
    
    def _detect_gesture(self, landmarks) -> str:
        """Detect hand gesture"""
        fingers = {
            'thumb': self._is_thumb_extended(landmarks),
            'index': self._is_finger_extended(landmarks, 8),
            'middle': self._is_finger_extended(landmarks, 12),
            'ring': self._is_finger_extended(landmarks, 16),
            'pinky': self._is_finger_extended(landmarks, 20)
        }
        
        extended_count = sum(fingers.values())
        
        if extended_count == 0:
            return 'fist'
        elif extended_count == 1 and fingers['index']:
            return 'point'
        elif extended_count == 2 and fingers['index'] and fingers['middle']:
            return 'peace'
        elif extended_count == 5:
            return 'open'
        else:
            return 'other'
    
    def draw_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw hand landmarks on frame"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        return frame
    
    def release(self):
        """Release resources"""
        self.hands.close()






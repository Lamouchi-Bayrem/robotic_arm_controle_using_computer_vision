
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class RoboticArm:
    """Simulates a robotic arm with inverse kinematics"""
    
    def __init__(self, num_joints: int = 6, link_lengths: List[float] = None):
        """
        Initialize robotic arm
        
        Args:
            num_joints: Number of joints
            link_lengths: Length of each link
        """
        self.num_joints = num_joints
        
        if link_lengths is None:
            # Default link lengths (normalized)
            self.link_lengths = [0.2, 0.15, 0.15, 0.1, 0.05, 0.03]
        else:
            self.link_lengths = link_lengths
        
        # Joint angles (initialized to zero)
        self.joint_angles = [0.0] * num_joints
        
        # Joint limits (in radians)
        self.joint_limits = [
            (-math.pi, math.pi),      # Base rotation
            (-math.pi/2, math.pi/2),  # Shoulder
            (-math.pi, math.pi),      # Elbow
            (-math.pi/2, math.pi/2),  # Wrist pitch
            (-math.pi/2, math.pi/2),  # Wrist roll
            (-math.pi/4, math.pi/4)   # Gripper rotation
        ]
        
        # Base position
        self.base_position = np.array([0.0, 0.0, 0.0])
        
        # Safety limits
        self.max_reach = sum(self.link_lengths)
        self.min_reach = 0.1
    
    def forward_kinematics(self, joint_angles: List[float] = None) -> Dict:
        """
        Calculate end effector position from joint angles
        
        Args:
            joint_angles: Joint angles (uses current if None)
            
        Returns:
            dict: End effector position and orientation
        """
        if joint_angles is None:
            joint_angles = self.joint_angles
        
        # Start from base
        position = np.array(self.base_position)
        rotation = np.eye(3)  # Identity matrix
        
        # Apply transformations for each joint
        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            # Rotation around Z axis (yaw)
            if i == 0:
                # Base rotation
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rotation = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
            else:
                # Joint rotation
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                # Rotate around Y axis (pitch)
                rot_y = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
                rotation = rotation @ rot_y
            
            # Move along link
            direction = rotation @ np.array([length, 0, 0])
            position = position + direction
        
        # Calculate orientation angles
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        pitch = math.asin(-rotation[2, 0])
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        
        return {
            'position': position.tolist(),
            'orientation': {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll)
            },
            'rotation_matrix': rotation.tolist()
        }
    
    def inverse_kinematics(self, target_position: np.ndarray,
                          target_orientation: Optional[Dict] = None) -> List[float]:
        """
        Calculate joint angles to reach target position
        
        Args:
            target_position: Target position [x, y, z]
            target_orientation: Target orientation (optional)
            
        Returns:
            list: Joint angles
        """
        target = np.array(target_position)
        
        # Check reachability
        distance = np.linalg.norm(target - self.base_position)
        if distance > self.max_reach:
            # Scale down to max reach
            direction = (target - self.base_position) / distance
            target = self.base_position + direction * self.max_reach
        elif distance < self.min_reach:
            # Scale up to min reach
            direction = (target - self.base_position) / (distance + 1e-6)
            target = self.base_position + direction * self.min_reach
        
        # Simplified IK for 3D arm
        # Base rotation (yaw)
        base_angle = math.atan2(target[1], target[0])
        
        # Project to XZ plane for arm calculation
        xz_distance = math.sqrt(target[0]**2 + target[1]**2)
        z_distance = target[2] - self.base_position[2]
        
        # Calculate shoulder and elbow angles (2D IK)
        l1, l2 = self.link_lengths[1], self.link_lengths[2]
        
        # Distance in XZ plane
        distance_2d = math.sqrt(xz_distance**2 + z_distance**2)
        
        if distance_2d > (l1 + l2):
            distance_2d = l1 + l2 - 0.01
        
        # Elbow angle (cosine law)
        cos_elbow = (l1**2 + l2**2 - distance_2d**2) / (2 * l1 * l2)
        cos_elbow = np.clip(cos_elbow, -1, 1)
        elbow_angle = math.acos(cos_elbow) - math.pi
        
        # Shoulder angle
        alpha = math.atan2(z_distance, xz_distance)
        beta = math.asin((l2 * math.sin(math.pi + elbow_angle)) / distance_2d)
        shoulder_angle = alpha - beta
        
        # Wrist angles (simplified - maintain orientation)
        wrist_pitch = -shoulder_angle - elbow_angle
        wrist_roll = 0.0
        
        # Gripper rotation
        gripper_rotation = 0.0
        
        angles = [
            base_angle,        # Base
            shoulder_angle,   # Shoulder
            elbow_angle,      # Elbow
            wrist_pitch,      # Wrist pitch
            wrist_roll,       # Wrist roll
            gripper_rotation  # Gripper
        ]
        
        # Apply joint limits
        angles = self._apply_joint_limits(angles)
        
        return angles
    
    def _apply_joint_limits(self, angles: List[float]) -> List[float]:
        """Apply joint limits to angles"""
        limited_angles = []
        for angle, (min_angle, max_angle) in zip(angles, self.joint_limits):
            limited_angles.append(np.clip(angle, min_angle, max_angle))
        return limited_angles
    
    def update_from_hand_control(self, control_signals: Dict) -> Dict:
        """
        Update arm position based on hand control signals
        
        Args:
            control_signals: Control signals from hand tracker
            
        Returns:
            dict: Update status and new joint angles
        """
        # Extract target position from hand
        rel_pos = control_signals['relative_position']
        
        # Scale relative position to arm workspace
        # Normalize to arm reach
        scale_factor = self.max_reach * 0.8  # Use 80% of max reach
        
        target_x = rel_pos['x'] * scale_factor
        target_y = rel_pos['y'] * scale_factor
        target_z = -rel_pos['z'] * scale_factor  # Invert Z (hand moves forward = arm moves forward)
        
        target_position = np.array([target_x, target_y, target_z])
        
        # Get orientation from hand
        hand_orientation = control_signals['orientation']
        target_orientation = {
            'yaw': hand_orientation['yaw'],
            'pitch': hand_orientation['pitch'],
            'roll': 0.0
        }
        
        # Calculate inverse kinematics
        new_angles = self.inverse_kinematics(target_position, target_orientation)
        
        # Smooth transition (interpolate)
        smoothing = 0.3  # 0 = no smoothing, 1 = no change
        self.joint_angles = [
            smoothing * old + (1 - smoothing) * new
            for old, new in zip(self.joint_angles, new_angles)
        ]
        
        # Get current end effector position
        fk_result = self.forward_kinematics()
        
        return {
            'success': True,
            'joint_angles': self.joint_angles.copy(),
            'end_effector': fk_result['position'],
            'target_reached': np.linalg.norm(
                np.array(fk_result['position']) - target_position
            ) < 0.05
        }
    
    def set_gripper(self, state: float):
        """
        Set gripper state
        
        Args:
            state: Gripper state (0 = closed, 1 = open)
        """
        # Gripper is typically the last joint or separate mechanism
        # For simulation, we can use a separate gripper state
        self.gripper_state = np.clip(state, 0, 1)
    
    def get_arm_state(self) -> Dict:
        """Get current arm state"""
        fk_result = self.forward_kinematics()
        return {
            'joint_angles': self.joint_angles.copy(),
            'end_effector_position': fk_result['position'],
            'end_effector_orientation': fk_result['orientation'],
            'gripper_state': getattr(self, 'gripper_state', 0.5),
            'link_lengths': self.link_lengths.copy(),
            'base_position': self.base_position.tolist()
        }






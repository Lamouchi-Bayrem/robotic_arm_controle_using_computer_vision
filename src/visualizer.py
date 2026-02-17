"""
Visualization Module
Visualizes robotic arm and hand tracking
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple
import io
from PIL import Image as PILImage


class ArmVisualizer:
    """Visualizes robotic arm in 3D"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.fig = None
        self.ax = None
    
    def plot_arm_3d(self, arm_state: Dict, hand_position: Dict = None) -> PILImage.Image:
        """
        Plot robotic arm in 3D
        
        Args:
            arm_state: Arm state from RoboticArm
            hand_position: Hand position for comparison (optional)
            
        Returns:
            PIL Image
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get arm configuration
        base_pos = np.array(arm_state['base_position'])
        joint_angles = arm_state['joint_angles']
        link_lengths = arm_state['link_lengths']
        
        # Calculate joint positions
        positions = [base_pos.copy()]
        current_pos = base_pos.copy()
        rotation = np.eye(3)
        
        for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
            if i == 0:
                # Base rotation
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
            else:
                # Joint rotation
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rot_y = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
                rotation = rotation @ rot_y
            
            # Move along link
            direction = rotation @ np.array([length, 0, 0])
            current_pos = current_pos + direction
            positions.append(current_pos.copy())
        
        # Plot arm links
        positions_array = np.array(positions)
        ax.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],
               'o-', linewidth=3, markersize=8, color='blue', label='Robotic Arm')
        
        # Plot joints
        for i, pos in enumerate(positions):
            ax.scatter(*pos, s=100, color='red' if i == 0 else 'orange')
            ax.text(pos[0], pos[1], pos[2], f'J{i}', fontsize=8)
        
        # Plot end effector
        end_effector = positions[-1]
        ax.scatter(*end_effector, s=200, color='green', marker='*', label='End Effector')
        
        # Plot hand position if provided
        if hand_position:
            hand_pos = np.array([
                hand_position.get('x', 0),
                hand_position.get('y', 0),
                hand_position.get('z', 0)
            ])
            ax.scatter(*hand_pos, s=200, color='purple', marker='^', label='Hand Target')
            # Draw line from end effector to hand
            ax.plot([end_effector[0], hand_pos[0]],
                   [end_effector[1], hand_pos[1]],
                   [end_effector[2], hand_pos[2]],
                   '--', color='gray', alpha=0.5, linewidth=1)
        
        # Set labels and limits
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robotic Arm 3D Visualization')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 0.5
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range * 2])
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close()
        
        return img
    
    def plot_arm_2d(self, arm_state: Dict, view: str = 'xy') -> PILImage.Image:
        """
        Plot robotic arm in 2D projection
        
        Args:
            arm_state: Arm state
            view: View plane ('xy', 'xz', 'yz')
            
        Returns:
            PIL Image
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate positions (same as 3D)
        base_pos = np.array(arm_state['base_position'])
        joint_angles = arm_state['joint_angles']
        link_lengths = arm_state['link_lengths']
        
        positions = [base_pos.copy()]
        current_pos = base_pos.copy()
        rotation = np.eye(3)
        
        for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
            if i == 0:
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rotation = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
            else:
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rot_y = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
                rotation = rotation @ rot_y
            
            direction = rotation @ np.array([length, 0, 0])
            current_pos = current_pos + direction
            positions.append(current_pos.copy())
        
        positions_array = np.array(positions)
        
        # Select axes based on view
        if view == 'xy':
            x_data = positions_array[:, 0]
            y_data = positions_array[:, 1]
            x_label, y_label = 'X (m)', 'Y (m)'
        elif view == 'xz':
            x_data = positions_array[:, 0]
            y_data = positions_array[:, 2]
            x_label, y_label = 'X (m)', 'Z (m)'
        else:  # yz
            x_data = positions_array[:, 1]
            y_data = positions_array[:, 2]
            x_label, y_label = 'Y (m)', 'Z (m)'
        
        # Plot arm
        ax.plot(x_data, y_data, 'o-', linewidth=3, markersize=8, color='blue')
        
        # Plot joints
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            ax.scatter(x, y, s=100, color='red' if i == 0 else 'orange')
            ax.text(x, y, f'J{i}', fontsize=8)
        
        # End effector
        ax.scatter(x_data[-1], y_data[-1], s=200, color='green', marker='*')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Robotic Arm - {view.upper()} View')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close()
        
        return img
    
    def draw_control_overlay(self, frame: np.ndarray, hand_result: Dict,
                            arm_state: Dict) -> np.ndarray:
        """
        Draw control overlay on camera frame
        
        Args:
            frame: Camera frame
            hand_result: Hand tracking result
            arm_state: Arm state
            
        Returns:
            numpy array: Annotated frame
        """
        annotated = frame.copy()
        
        if hand_result['hand_detected']:
            # Draw hand landmarks
            # (This would be done by hand tracker, but we can add additional info)
            
            # Draw control info
            control = hand_result['control_signals']
            target = control['target_position']
            
            # Draw target point
            h, w = frame.shape[:2]
            target_x = int(target['x'] * w)
            target_y = int(target['y'] * h)
            cv2.circle(annotated, (target_x, target_y), 10, (0, 255, 0), -1)
            cv2.putText(annotated, "Target", (target_x + 15, target_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw gripper state
            gripper_state = control['gripper']
            gripper_text = f"Gripper: {gripper_state:.2f}"
            cv2.putText(annotated, gripper_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw gesture
            gesture = hand_result['gesture']
            cv2.putText(annotated, f"Gesture: {gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return annotated






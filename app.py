import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import time

from src.hand_tracker import HandTracker
from src.robotic_arm import RoboticArm
from src.visualizer import ArmVisualizer




# Page configuration
st.set_page_config(
    page_title="Robotic Arm Control",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'hand_tracker' not in st.session_state:
    st.session_state.hand_tracker = HandTracker()
if 'robotic_arm' not in st.session_state:
    st.session_state.robotic_arm = RoboticArm(num_joints=6)
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = ArmVisualizer()
if 'control_active' not in st.session_state:
    st.session_state.control_active = False
if 'arm_history' not in st.session_state:
    st.session_state.arm_history = []


def main():
    """Main application"""
    st.title("🤖 Robotic Arm Control via Hand Simulation")
    st.markdown("**Control a robotic arm using your hand movements and gestures**")
    
    # Safety warning
    st.warning(
        "⚠️ **SAFETY WARNING**: This is a simulation tool. For real robotic arm control, "
        "ensure proper safety measures, emergency stops, and workspace limits are in place."
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Settings")
        
        # Control mode
        st.subheader("🎮 Control Mode")
        control_mode = st.radio(
            "Mode",
            ["Position Control", "Gesture Control", "Manual Control"],
            index=0
        )
        
        # Safety settings
        st.subheader("🛡️ Safety Settings")
        enable_safety = st.checkbox("Enable Safety Limits", value=True)
        max_speed = st.slider("Max Speed", 0.1, 2.0, 1.0, 0.1)
        smoothing = st.slider("Motion Smoothing", 0.0, 1.0, 0.3, 0.1)
        
        # Arm configuration
        st.subheader("🔧 Arm Configuration")
        num_joints = st.number_input("Number of Joints", min_value=3, max_value=7, value=6)
        
        if st.button("🔄 Reset Arm"):
            st.session_state.robotic_arm = RoboticArm(num_joints=num_joints)
            st.session_state.arm_history.clear()
            st.success("Arm reset!")
        
        # Visualization options
        st.subheader("📊 Visualization")
        show_3d = st.checkbox("Show 3D View", value=True)
        show_2d = st.checkbox("Show 2D Projections", value=False)
        view_plane = st.selectbox("2D View Plane", ["xy", "xz", "yz"])
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        1. Start camera feed
        2. Enable control
        3. Move hand to control arm
        4. Use gestures:
           - Point: Move end effector
           - Fist: Stop movement
           - Open: Open gripper
           - Peace: Close gripper
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed & Hand Tracking")
        
        # Control toggle
        if st.button("🎥 Start Control" if not st.session_state.control_active else "🛑 Stop Control"):
            st.session_state.control_active = not st.session_state.control_active
            if st.session_state.control_active:
                st.session_state.arm_history.clear()
        
        if st.session_state.control_active:
            camera_input = st.camera_input("Show your hand", key="arm_control_camera")
            
            if camera_input:
                # Convert to OpenCV format
                img = Image.open(camera_input)
                frame = np.array(img)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Track hand
                hand_result = st.session_state.hand_tracker.track_hand(frame_bgr)
                
                # Draw hand landmarks
                if hand_result['hand_detected']:
                    frame_bgr = st.session_state.hand_tracker.draw_landmarks(
                        frame_bgr, hand_result['landmarks']
                    )
                    
                    # Update robotic arm based on hand
                    if control_mode == "Position Control":
                        update_result = st.session_state.robotic_arm.update_from_hand_control(
                            hand_result['control_signals']
                        )
                        
                        # Update gripper
                        gripper_state = hand_result['control_signals']['gripper']
                        st.session_state.robotic_arm.set_gripper(gripper_state)
                        
                        # Store history
                        arm_state = st.session_state.robotic_arm.get_arm_state()
                        st.session_state.arm_history.append({
                            'timestamp': datetime.now(),
                            'arm_state': arm_state,
                            'hand_control': hand_result['control_signals']
                        })
                        
                        # Keep only last 100 entries
                        if len(st.session_state.arm_history) > 100:
                            st.session_state.arm_history.pop(0)
                    
                    # Draw control overlay
                    frame_bgr = st.session_state.visualizer.draw_control_overlay(
                        frame_bgr, hand_result, st.session_state.robotic_arm.get_arm_state()
                    )
                else:
                    st.warning("No hand detected. Show your hand to the camera.")
                
                # Display frame
                st.image(frame_bgr, use_container_width=True)
                
                # Arm visualization
                if hand_result['hand_detected']:
                    arm_state = st.session_state.robotic_arm.get_arm_state()
                    
                    # Get hand target for visualization
                    hand_target = hand_result['control_signals']['target_position']
                    
                    if show_3d:
                        st.markdown("### 🤖 3D Arm Visualization")
                        arm_3d = st.session_state.visualizer.plot_arm_3d(
                            arm_state, hand_target
                        )
                        st.image(arm_3d, use_container_width=True)
                    
                    if show_2d:
                        st.markdown(f"### 📐 2D {view_plane.upper()} View")
                        arm_2d = st.session_state.visualizer.plot_arm_2d(arm_state, view_plane)
                        st.image(arm_2d, use_container_width=True)
        else:
            st.info("Click 'Start Control' to begin hand tracking and arm control")
    
    with col2:
        st.subheader("📊 Arm Status")
        
        if st.session_state.control_active:
            arm_state = st.session_state.robotic_arm.get_arm_state()
            
            # Current position
            st.markdown("**End Effector Position:**")
            ee_pos = arm_state['end_effector_position']
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("X", f"{ee_pos[0]:.3f}m")
            with col_b:
                st.metric("Y", f"{ee_pos[1]:.3f}m")
            with col_c:
                st.metric("Z", f"{ee_pos[2]:.3f}m")
            
            # Joint angles
            st.markdown("**Joint Angles:**")
            joint_angles = arm_state['joint_angles']
            angles_df = pd.DataFrame({
                'Joint': [f'J{i}' for i in range(len(joint_angles))],
                'Angle (rad)': [f"{angle:.3f}" for angle in joint_angles],
                'Angle (deg)': [f"{np.degrees(angle):.1f}°" for angle in joint_angles]
            })
            st.dataframe(angles_df, use_container_width=True, hide_index=True)
            
            # Gripper state
            st.markdown("**Gripper:**")
            gripper_state = arm_state['gripper_state']
            st.progress(gripper_state)
            st.text(f"State: {gripper_state:.2f} ({'Open' if gripper_state > 0.5 else 'Closed'})")
            
            # Control signals (if hand detected)
            if st.session_state.arm_history:
                latest = st.session_state.arm_history[-1]
                st.markdown("**Latest Control Signals:**")
                control = latest['hand_control']
                
                st.json({
                    'target_position': control['target_position'],
                    'gripper': f"{control['gripper']:.2f}",
                    'gesture': st.session_state.hand_tracker._detect_gesture(
                        latest.get('hand_landmarks', None)
                    ) if latest.get('hand_landmarks') else 'Unknown'
                })
        else:
            st.info("Start control to see arm status")
        
        # History
        if st.session_state.arm_history:
            st.subheader("📈 Movement History")
            st.metric("Data Points", len(st.session_state.arm_history))
            
            if st.button("📥 Export History"):
                history_df = pd.DataFrame([
                    {
                        'timestamp': entry['timestamp'],
                        'x': entry['arm_state']['end_effector_position'][0],
                        'y': entry['arm_state']['end_effector_position'][1],
                        'z': entry['arm_state']['end_effector_position'][2],
                        'gripper': entry['arm_state']['gripper_state']
                    }
                    for entry in st.session_state.arm_history
                ])
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"arm_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()






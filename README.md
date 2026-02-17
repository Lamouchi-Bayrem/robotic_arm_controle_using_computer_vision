# Robotic Arm Control via Hand Simulation

An AI-powered system that controls a robotic arm using human hand tracking and gesture recognition. Uses computer vision to track hand movements and translates them into robotic arm commands with inverse kinematics.

## Features

- ✅ **Hand Tracking**: Real-time hand pose estimation using MediaPipe
- ✅ **Inverse Kinematics**: Automatic joint angle calculation for target positions
- ✅ **Forward Kinematics**: Calculate end effector position from joint angles
- ✅ **Gesture Control**: Control arm with hand gestures (point, fist, open, peace)
- ✅ **Gripper Control**: Open/close gripper based on thumb-index distance
- ✅ **3D Visualization**: Real-time 3D visualization of robotic arm
- ✅ **2D Projections**: Multiple 2D views (XY, XZ, YZ planes)
- ✅ **Safety Limits**: Joint limits and workspace constraints
- ✅ **Motion Smoothing**: Smooth transitions between positions
- ✅ **History Tracking**: Record and export movement history

## Control Modes

### 1. Position Control
- Move hand to control end effector position
- Real-time tracking and arm movement
- Automatic inverse kinematics

### 2. Gesture Control
- **Point**: Move end effector to finger tip
- **Fist**: Stop movement
- **Open Hand**: Open gripper
- **Peace Sign**: Close gripper

### 3. Manual Control
- Direct joint angle control (future feature)

## Requirements

- Python 3.8+
- Webcam/Camera
- Modern web browser

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```
   
   Or use the run script:
   ```bash
   python run.py
   ```

3. **Open in browser**: The app will automatically open at `http://localhost:8501`

## Usage

### Basic Control

1. **Start Control**:
   - Click "Start Control" button
   - Allow camera permissions
   - Show your hand to the camera

2. **Control Arm**:
   - Move your hand to control end effector position
   - Index finger tip determines target position
   - Hand orientation controls arm orientation

3. **Control Gripper**:
   - Bring thumb and index finger together to close gripper
   - Spread them apart to open gripper

4. **View Visualization**:
   - See 3D arm visualization
   - Check 2D projections
   - Monitor joint angles and positions

### Gestures

- **Point (Index Extended)**: Move end effector to finger tip
- **Fist (All Closed)**: Stop/Freeze arm movement
- **Open Hand (All Extended)**: Open gripper fully
- **Peace Sign (Index + Middle)**: Close gripper

## Technical Details

### Hand Tracking
- **MediaPipe Hands**: 21 hand landmarks
- **Control Signals**: Position, orientation, gripper state
- **Gesture Recognition**: Finger extension detection

### Robotic Arm
- **6-DOF Arm**: 6 joints (base, shoulder, elbow, wrist pitch, wrist roll, gripper)
- **Inverse Kinematics**: Analytical IK for 3D positioning
- **Forward Kinematics**: Calculate end effector from joint angles
- **Joint Limits**: Safety constraints on joint angles

### Kinematics

**Forward Kinematics**:
- Calculate end effector position from joint angles
- Uses rotation matrices and link transformations

**Inverse Kinematics**:
- Calculate joint angles to reach target position
- Simplified 2D IK projected to 3D
- Applies joint limits and workspace constraints

## Project Structure

```
robotic_arm_control/
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py    # Hand tracking and control signal extraction
│   ├── robotic_arm.py      # Arm simulation and kinematics
│   └── visualizer.py       # 3D/2D visualization
├── app.py                   # Streamlit main app
├── run.py                   # Entry point
├── requirements.txt
└── README.md
```

## Safety Features

⚠️ **Important Safety Considerations**:

- **Joint Limits**: Prevents arm from exceeding safe angles
- **Workspace Limits**: Constrains end effector to safe region
- **Speed Limiting**: Maximum movement speed control
- **Motion Smoothing**: Prevents sudden jerky movements
- **Emergency Stop**: Fist gesture stops movement

**For Real Robotic Arms**:
- Always implement hardware emergency stops
- Use proper safety interlocks
- Set workspace boundaries
- Monitor for collisions
- Have trained operator supervision

## Use Cases

- **Education**: Learn robotics and inverse kinematics
- **Prototyping**: Test control algorithms before hardware
- **Teleoperation**: Remote control of robotic arms
- **Rehabilitation**: Assistive robotics for therapy
- **Research**: Human-robot interaction studies
- **Entertainment**: Interactive robotic demonstrations

## Limitations

- **Simulation Only**: This is a simulation tool, not real hardware control
- **Simplified IK**: Uses simplified inverse kinematics (may not be optimal)
- **Hand Tracking**: Requires good lighting and clear hand visibility
- **Latency**: Real-time control depends on processing speed
- **Accuracy**: Hand tracking accuracy affects control precision

## Future Enhancements

- [ ] Real hardware integration (ROS, Arduino, etc.)
- [ ] Advanced IK algorithms (iterative, neural network-based)
- [ ] Collision detection and avoidance
- [ ] Trajectory planning
- [ ] Force feedback simulation
- [ ] Multi-hand control
- [ ] Machine learning for gesture recognition
- [ ] Haptic feedback integration
- [ ] Mobile app version
- [ ] VR/AR integration

## Troubleshooting

### Hand Not Detected
- Ensure good lighting
- Keep hand clearly visible
- Remove background clutter
- Try different hand positions

### Arm Not Moving
- Check that control is active
- Verify hand is being tracked
- Check safety limits
- Ensure camera is working

### Inaccurate Control
- Calibrate hand position
- Improve lighting conditions
- Check camera angle
- Adjust smoothing parameters

### Performance Issues
- Close other applications
- Reduce visualization complexity
- Lower camera resolution
- Disable unnecessary features

## License

This project is provided as-is for educational and portfolio purposes.

## Acknowledgments

- MediaPipe for hand tracking
- OpenCV for computer vision
- Streamlit for web framework
- Robotics research community

## Disclaimer

This is a simulation tool for educational purposes. For real robotic arm control, ensure proper safety measures, emergency stops, and workspace limits are implemented. Always follow manufacturer safety guidelines and local regulations.





#

import cv2
import mediapipe as mp
import time
import numpy as np
import webbrowser
from datetime import datetime

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Custom drawing specs for better visibility
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))
connection_spec = mp_drawing.DrawingSpec(thickness=2, color=(255, 255, 0))

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Create transparent overlay
def create_overlay(frame, alpha=0.4):
    overlay = frame.copy()
    output = frame.copy()
    
    # Create semi-transparent overlay for the sidebar
    cv2.rectangle(overlay, (0, 0), (300, frame.shape[0]), (50, 50, 50), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

# Tutorial function with improved UI
def show_tutorial_func():
    tutorial_text = [
        "HOW TO PERFORM A PROPER BICEP CURL:",
        "1. Stand with feet shoulder-width apart.",
        "2. Keep your elbows close to your torso.",
        "3. Hold weights with palms facing forward.",
        "4. Curl the weights toward your shoulders.",
        "5. Lower slowly back to starting position."
    ]
    
    # Create a nicer looking tutorial window
    tutorial_window = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    # Add a gradient background (blue-themed)
    for i in range(800):
        # Blue gradient
        color = (60 + i//40, 30 + i//40, 20 + i//40)
        cv2.line(tutorial_window, (0, i), (1200, i), color, 1)
    
    window = tutorial_window.copy()
    
    # Add title bar (dark blue)
    cv2.rectangle(window, (0, 0), (1200, 100), (70, 45, 40), -1)
    cv2.putText(window, "BICEP CURL TUTORIAL", (350, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    
    # Display tutorial text with spacing
    for i, line in enumerate(tutorial_text):
        y_pos = 200 + i*70  # Vertical spacing
        cv2.putText(window, line, (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (120, 255, 255), 2)
    
    # Add button outlines
    cv2.rectangle(window, (300, 650), (550, 720), (255, 120, 0), 2)
    cv2.rectangle(window, (650, 650), (900, 720), (0, 200, 100), 2)
    cv2.putText(window, "TUTORIAL (T)", (335, 695), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 120, 0), 2)
    cv2.putText(window, "START (C)", (690, 695), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 100), 2)
    
    cv2.imshow("Bicep Curl Tutorial", window)
    
    # Wait for key press
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('t'):
            webbrowser.open("https://www.youtube.com/watch?v=ykJmrZ5v0Oo")
            break
    
    cv2.destroyWindow("Bicep Curl Tutorial")

# Improved settings menu
def settings_menu():
    # Increased size to avoid text overlap
    settings_window = np.zeros((900, 1200, 3), dtype=np.uint8)
    
    # Default values
    cam_index = 0
    sets = 3
    reps_left = 10
    reps_right = 10
    rest_time = 30
    show_tutorial = True
    selected = 0  # Currently selected option
    
    options = ["Camera Index", "Number of Sets", "Left Arm Reps", "Right Arm Reps", "Rest Time (sec)", "Show Tutorial", "Start Workout"]
    values = [cam_index, sets, reps_left, reps_right, rest_time, "Yes", ""]
    
    while True:
        window = settings_window.copy()
        
        # Add a blue gradient background
        for i in range(900):
            # Blue gradient (dark to medium)
            color = (60 + i//45, 40 + i//45, 20 + i//30)
            cv2.line(window, (0, i), (1200, i), color, 1)
        
        # Add title bar (darker blue)
        cv2.rectangle(window, (0, 0), (1200, 120), (50, 30, 40), -1)
        cv2.putText(window, "BICEP CURL SETTINGS", (300, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Display options with increased spacing
        for i, (option, value) in enumerate(zip(options, values)):
            y_pos = 250 + i*100  # Vertical spacing
            
            # Highlight selected option
            color = (255, 255, 0) if i == selected else (255, 255, 255)
            
            # Left align labels
            cv2.putText(window, f"{option}:", (150, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw box for the value
            if i < len(options) - 1:  # Not for the "Start" button
                # Create value box with more space
                cv2.rectangle(window, (650, y_pos-40), (950, y_pos+20), 
                              (100, 100, 100) if i != selected else (50, 150, 200), 2)
                cv2.putText(window, f"{value}", (700, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            else:
                # Make "Start Workout" a green button
                cv2.rectangle(window, (400, y_pos-40), (800, y_pos+20), 
                              (0, 200, 0) if i == selected else (0, 150, 0), -1)
                cv2.putText(window, "START WORKOUT", (450, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Instructions
        cv2.putText(window, "Use UP/DOWN to navigate, LEFT/RIGHT to change values", 
                    (200, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.putText(window, "Press ENTER to confirm selection", 
                    (200, 850), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        cv2.imshow("Settings", window)
        
        key = cv2.waitKey(100) & 0xFF
        
        # Navigation
        if key == 82 or key == ord('w'):  # Up arrow or W
            selected = max(0, selected - 1)
        elif key == 84 or key == ord('s'):  # Down arrow or S
            selected = min(len(options) - 1, selected + 1)
        elif key == 83 or key == ord('d'):  # Right arrow or D
            if selected == 0:
                values[selected] = min(5, values[selected] + 1)
            elif selected == 1:
                values[selected] = min(10, values[selected] + 1)
            elif selected == 2:
                values[selected] = min(20, values[selected] + 1)
            elif selected == 3:
                values[selected] = min(20, values[selected] + 1)
            elif selected == 4:
                values[selected] = min(60, values[selected] + 5)
            elif selected == 5:
                values[selected] = "No" if values[selected] == "Yes" else "Yes"
        elif key == 81 or key == ord('a'):  # Left arrow or A
            if selected == 0:
                values[selected] = max(0, values[selected] - 1)
            elif selected == 1:
                values[selected] = max(1, values[selected] - 1)
            elif selected == 2:
                values[selected] = max(1, values[selected] - 1)
            elif selected == 3:
                values[selected] = max(1, values[selected] - 1)
            elif selected == 4:
                values[selected] = max(5, values[selected] - 5)
            elif selected == 5:
                values[selected] = "No" if values[selected] == "Yes" else "Yes"
        elif key == 13:  # Enter key
            if selected == 6:  # Start Workout button
                break
        elif key == 27:  # ESC key
            cv2.destroyWindow("Settings")
            exit()
    
    cv2.destroyWindow("Settings")
    
    # Parse values
    cam_index = values[0]
    sets = values[1]
    reps_left = values[2]
    reps_right = values[3]
    rest_time = values[4]
    tutorial_choice = values[5]
    
    return cam_index, sets, reps_left, reps_right, rest_time, tutorial_choice == "Yes"

# Get workout statistics display for bicep curls
def draw_stats(frame, left_counter, right_counter, reps_left, reps_right, current_set, sets, arm_mode="Both", form_msg="", angle=0):
    frame_height, frame_width, _ = frame.shape
    stats_frame = create_overlay(frame)
    
    # Progress bars for current set
    # Left arm progress
    left_progress = int((left_counter / reps_left) * 260)
    cv2.rectangle(stats_frame, (20, 80), (280, 110), (100, 100, 100), -1)
    cv2.rectangle(stats_frame, (20, 80), (20 + left_progress, 110), (0, 255, 0), -1)
    
    # Right arm progress
    right_progress = int((right_counter / reps_right) * 260)
    cv2.rectangle(stats_frame, (20, 140), (280, 170), (100, 100, 100), -1)
    cv2.rectangle(stats_frame, (20, 140), (20 + right_progress, 170), (0, 255, 0), -1)
    
    # Text stats
    cv2.putText(stats_frame, f"SET: {current_set+1}/{sets}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(stats_frame, f"RIGHT ARM: {left_counter}/{reps_left}", (20, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(stats_frame, f"LEFT ARM: {right_counter}/{reps_right}", (20, 130), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Current mode
    cv2.putText(stats_frame, f"MODE: {arm_mode}", (20, 190), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
    
    # Current time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(stats_frame, current_time, (frame_width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Form feedback
    if form_msg:
        color = (0, 255, 0) if "Good" in form_msg else (0, 0, 255)
        cv2.putText(stats_frame, form_msg, (20, 220), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    
    # Angle display
    if angle > 0:
        cv2.putText(stats_frame, f"Arm Angle: {int(angle)}", (20, 260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    
    # Instructions
    cv2.putText(stats_frame, "Press 'Q' to quit, 'M' to change mode", (20, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return stats_frame

# Function to show countdown
def show_countdown(cap, message, seconds, color=(0, 255, 255)):
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            continue
            
        overlay = frame.copy()
        # Add semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
        countdown_frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add countdown message
        cv2.putText(countdown_frame, f"{message}", (frame.shape[1]//2 - 200, frame.shape[0]//2 - 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        cv2.putText(countdown_frame, f"{i}", (frame.shape[1]//2, frame.shape[0]//2 + 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 3, color, 4)
        
        cv2.imshow("Bicep Curl Counter", countdown_frame)
        cv2.waitKey(1000)

# Main program
def main():
    # Get settings from UI menu
    cam_index, sets, reps_left, reps_right, rest_time, show_tutorial = settings_menu()
    
    if show_tutorial:
        show_tutorial_func()
    
    # Start workout
    cap = cv2.VideoCapture(cam_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Get camera properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare window
    cv2.namedWindow("Bicep Curl Counter", cv2.WINDOW_NORMAL)
    
    # Arm mode cycling (Both, Left, Right)
    arm_modes = ["Both", "Left Arm", "Right Arm"]
    current_mode_index = 0
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        # Show ready message
        show_countdown(cap, "GET READY!", 3, color=(0, 255, 255))
        
        for current_set in range(sets):
            left_counter = 0
            right_counter = 0
            left_stage = None
            right_stage = None
            
            # Countdown before set
            show_countdown(cap, f"SET {current_set+1} STARTS IN", 3)
            
            while left_counter < reps_left or right_counter < reps_right:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to receive frame from camera")
                    break
                
                # Flip the frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                form_msg = ""
                left_angle = 0
                right_angle = 0
                
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates for left arm
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Get coordinates for right arm
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    # Calculate angles for both arms
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    # Visualize angles on the frame
                    cv2.putText(image, f"{int(left_angle)}", 
                                (int(left_elbow[0] * frame_width + 10), int(left_elbow[1] * frame_height)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, f"{int(right_angle)}", 
                                (int(right_elbow[0] * frame_width + 10), int(right_elbow[1] * frame_height)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Form feedback based on current mode
                    current_mode = arm_modes[current_mode_index]
                    
                    # Left arm rep counting
                    if current_mode in ["Both", "Left Arm"] and left_counter < reps_left:
                        if left_angle > 160:
                            left_stage = "down"
                            if current_mode == "Left Arm":
                                form_msg = "Arm Extended"
                        if left_angle < 50 and left_stage == "down":
                            left_stage = "up"
                            left_counter += 1
                            form_msg = "Good Curl!"
                            # Visual feedback for rep completion
                            cv2.rectangle(image, (0, 0), (frame_width//2, frame_height), (0, 255, 0), 10)
                    
                    # Right arm rep counting
                    if current_mode in ["Both", "Right Arm"] and right_counter < reps_right:
                        if right_angle > 160:
                            right_stage = "down"
                            if current_mode == "Right Arm":
                                form_msg = "Arm Extended"
                        if right_angle < 50 and right_stage == "down":
                            right_stage = "up"
                            right_counter += 1
                            form_msg = "Good Curl!"
                            # Visual feedback for rep completion
                            cv2.rectangle(image, (frame_width//2, 0), (frame_width, frame_height), (0, 255, 0), 10)
                    
                    # Overall form feedback
                    if current_mode == "Both":
                        if (left_angle < 50 and right_angle > 120) or (right_angle < 50 and left_angle > 120):
                            form_msg = "Keep Arms in Sync"
                        elif left_angle < 90 and right_angle < 90:
                            form_msg = "Good Form!"
                    
                except Exception as e:
                    pass
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        drawing_spec,
                        connection_spec
                    )
                
                # Get current angle for display based on mode
                display_angle = 0
                if current_mode == "Left Arm":
                    display_angle = left_angle
                elif current_mode == "Right Arm":
                    display_angle = right_angle
                elif current_mode == "Both":
                    display_angle = (left_angle + right_angle) / 2
                
                # Draw workout statistics
                stats_image = draw_stats(image, left_counter, right_counter, reps_left, reps_right, 
                                          current_set, sets, arm_modes[current_mode_index], form_msg, display_angle)
                
                # Show the frame
                cv2.imshow("Bicep Curl Counter", stats_image)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    return
                elif key == ord('m'):
                    # Cycle through arm modes
                    current_mode_index = (current_mode_index + 1) % len(arm_modes)
                
                # Check if all reps are completed for this set
                if left_counter >= reps_left and right_counter >= reps_right:
                    break
            
            # Set completed animation
            completed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(completed_frame, f"SET {current_set+1} COMPLETED!", (frame_width//2 - 200, frame_height//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Bicep Curl Counter", completed_frame)
            cv2.waitKey(1000)
            
            # Skip rest period after the last set
            if current_set < sets - 1:
                show_countdown(cap, f"REST TIME", rest_time, color=(100, 200, 255))
    
    # Workout complete animation
    for _ in range(3):  # Flash effect
        completed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.putText(completed_frame, "WORKOUT COMPLETE!", (frame_width//2 - 200, frame_height//2 - 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(completed_frame, f"Total: {sets} sets × L:{reps_left}/R:{reps_right} reps", 
                    (frame_width//2 - 230, frame_height//2 + 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Bicep Curl Counter", completed_frame)
        cv2.waitKey(300)
        
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.imshow("Bicep Curl Counter", blank_frame)
        cv2.waitKey(200)
    
    # Final message
    final_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.putText(final_frame, "WORKOUT COMPLETE!", (frame_width//2 - 200, frame_height//2 - 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(final_frame, f"Total: {sets} sets × L:{reps_left}/R:{reps_right} reps", 
                (frame_width//2 - 230, frame_height//2 + 30), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_frame, "Press any key to exit", (frame_width//2 - 150, frame_height//2 + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.imshow("Bicep Curl Counter", final_frame)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
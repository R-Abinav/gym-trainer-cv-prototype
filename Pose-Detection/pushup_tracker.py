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
        "HOW TO PERFORM A PROPER PUSHUP:",
        "1. Start in a plank position with arms shoulder-width apart.",
        "2. Keep your body in a straight line from head to heels.",
        "3. Lower your chest until arms form 90-degree angles.",
        "4. Push through your palms to return to starting position.",
        "5. Keep core engaged throughout the movement."
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
    cv2.putText(window, "PUSHUP FORM TUTORIAL", (350, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    
    # Display tutorial text with good spacing
    for i, line in enumerate(tutorial_text):
        y_pos = 200 + i*70  # Increased vertical spacing
        cv2.putText(window, line, (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (120, 255, 255), 2)
    
    # Add button outlines
    cv2.rectangle(window, (300, 650), (550, 720), (255, 120, 0), 2)
    cv2.rectangle(window, (650, 650), (900, 720), (0, 200, 100), 2)
    cv2.putText(window, "TUTORIAL (T)", (335, 695), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 120, 0), 2)
    cv2.putText(window, "START (C)", (690, 695), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 100), 2)
    
    cv2.imshow("Pushup Tutorial", window)
    
    # Wait for key press
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('t'):
            webbrowser.open("https://www.youtube.com/watch?v=IODxDxX7oi4")
            break
    
    cv2.destroyWindow("Pushup Tutorial")

# Settings menu
def settings_menu():
    # Create large window to avoid text overlap
    settings_window = np.zeros((900, 1200, 3), dtype=np.uint8)
    
    # Default values
    cam_index = 0
    sets = 3
    reps_per_set = 10
    rest_time = 30
    show_tutorial = True
    selected = 0  # Currently selected option
    
    options = ["Camera Index", "Number of Sets", "Reps per Set", "Rest Time (sec)", "Show Tutorial", "Start Workout"]
    values = [cam_index, sets, reps_per_set, rest_time, "Yes", ""]
    
    while True:
        window = settings_window.copy()
        
        # Add a blue gradient background
        for i in range(900):
            # Blue gradient (dark to medium)
            color = (60 + i//45, 40 + i//45, 20 + i//30)
            cv2.line(window, (0, i), (1200, i), color, 1)
        
        # Add title bar
        cv2.rectangle(window, (0, 0), (1200, 120), (50, 30, 40), -1)
        cv2.putText(window, "PUSHUP COUNTER SETTINGS", (250, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Display options with good spacing
        for i, (option, value) in enumerate(zip(options, values)):
            y_pos = 250 + i*100  # More vertical spacing
            
            # Highlight selected option
            color = (255, 255, 0) if i == selected else (255, 255, 255)
            
            # Left align labels
            cv2.putText(window, f"{option}:", (150, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw box for the value
            if i < len(options) - 1:  # Not for the "Start" button
                # Create value box
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
                values[selected] = min(60, values[selected] + 5)
            elif selected == 4:
                values[selected] = "No" if values[selected] == "Yes" else "Yes"
        elif key == 81 or key == ord('a'):  # Left arrow or A
            if selected == 0:
                values[selected] = max(0, values[selected] - 1)
            elif selected == 1:
                values[selected] = max(1, values[selected] - 1)
            elif selected == 2:
                values[selected] = max(1, values[selected] - 1)
            elif selected == 3:
                values[selected] = max(5, values[selected] - 5)
            elif selected == 4:
                values[selected] = "No" if values[selected] == "Yes" else "Yes"
        elif key == 13:  # Enter key
            if selected == 5:  # Start Workout button
                break
        elif key == 27:  # ESC key
            cv2.destroyWindow("Settings")
            exit()
    
    cv2.destroyWindow("Settings")
    
    # Parse values
    cam_index = values[0]
    sets = values[1]
    reps_per_set = values[2]
    rest_time = values[3]
    tutorial_choice = values[4]
    
    return cam_index, sets, reps_per_set, rest_time, tutorial_choice == "Yes"

# Get workout statistics display
def draw_stats(frame, counter, reps_per_set, current_set, sets, form_msg="", angle=0):
    frame_height, frame_width, _ = frame.shape
    stats_frame = create_overlay(frame)
    
    # Progress bar for current set
    progress = int((counter / reps_per_set) * 260)
    cv2.rectangle(stats_frame, (20, 80), (280, 110), (100, 100, 100), -1)
    cv2.rectangle(stats_frame, (20, 80), (20 + progress, 110), (0, 255, 0), -1)
    
    # Text stats
    cv2.putText(stats_frame, f"SET: {current_set+1}/{sets}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(stats_frame, f"REPS: {counter}/{reps_per_set}", (20, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Current time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(stats_frame, current_time, (frame_width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Form feedback
    if form_msg:
        color = (0, 255, 0) if "Good" in form_msg else (0, 0, 255)
        cv2.putText(stats_frame, form_msg, (20, 150), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    
    # Angle display
    cv2.putText(stats_frame, f"Elbow Angle: {int(angle)}", (20, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    
    # Instructions
    cv2.putText(stats_frame, "Press 'Q' to quit", (20, frame_height - 20), 
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
        
        cv2.imshow("Pushup Counter", countdown_frame)
        cv2.waitKey(1000)

# Main program
def main():
    # Get settings from UI menu
    cam_index, sets, reps_per_set, rest_time, show_tutorial = settings_menu()
    
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
    cv2.namedWindow("Pushup Counter", cv2.WINDOW_NORMAL)
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        # Show ready message
        show_countdown(cap, "GET READY!", 3, color=(0, 255, 255))
        
        for current_set in range(sets):
            counter = 0
            stage = None
            
            # Countdown before set
            show_countdown(cap, f"SET {current_set+1} STARTS IN", 3)
            
            start_time = time.time()
            
            while counter < reps_per_set:
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
                angle = 0
                
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
                    
                    # Use the average angle
                    angle = (left_angle + right_angle) / 2
                    
                    # Check for proper body alignment (back straight)
                    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                          landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    # Calculate hip position relative to shoulders and ankles to check body alignment
                    hip_y = (left_hip[1] + right_hip[1]) / 2
                    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                    
                    # Visualize angles on the frame
                    cv2.putText(image, f"{int(left_angle)}", 
                                (int(left_elbow[0] * frame_width + 10), int(left_elbow[1] * frame_height)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image, f"{int(right_angle)}", 
                                (int(right_elbow[0] * frame_width + 10), int(right_elbow[1] * frame_height)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Form feedback
                    if angle < 70:
                        form_msg = "Good Depth!"
                    elif angle > 150:
                        form_msg = "Arms Extended"
                    
                    # Check if the back is straight
                    if abs(hip_y - ((shoulder_y + ankle_y) / 2)) > 0.1:
                        form_msg = "Keep Body Straight!"
                    
                    # Rep counting logic
                    if angle > 160:
                        stage = "up"
                    if angle < 90 and stage == "up":
                        stage = "down"
                        counter += 1
                        # Play success sound or visual feedback
                        cv2.rectangle(image, (0, 0), (frame_width, frame_height), (0, 255, 0), 10)
                        
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
                
                # Draw workout statistics
                stats_image = draw_stats(image, counter, reps_per_set, current_set, sets, form_msg, angle)
                
                # Show the frame
                cv2.imshow("Pushup Counter", stats_image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    return
            
            # Set completed animation
            completed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(completed_frame, f"SET {current_set+1} COMPLETED!", (frame_width//2 - 200, frame_height//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Pushup Counter", completed_frame)
            cv2.waitKey(1000)
            
            # Skip rest period after the last set
            if current_set < sets - 1:
                show_countdown(cap, f"REST TIME", rest_time, color=(100, 200, 255))
    
    # Workout complete animation
    for _ in range(3):  # Flash effect
        completed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.putText(completed_frame, "WORKOUT COMPLETE!", (frame_width//2 - 200, frame_height//2 - 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(completed_frame, f"Total: {sets} sets × {reps_per_set} reps", 
                    (frame_width//2 - 180, frame_height//2 + 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Pushup Counter", completed_frame)
        cv2.waitKey(300)
        
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.imshow("Pushup Counter", blank_frame)
        cv2.waitKey(200)
    
    # Final message
    final_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.putText(final_frame, "WORKOUT COMPLETE!", (frame_width//2 - 200, frame_height//2 - 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(final_frame, f"Total: {sets} sets × {reps_per_set} reps", 
                (frame_width//2 - 180, frame_height//2 + 30), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_frame, "Press any key to exit", (frame_width//2 - 150, frame_height//2 + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.imshow("Pushup Counter", final_frame)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
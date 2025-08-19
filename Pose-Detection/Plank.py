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
        "HOW TO PERFORM A PROPER PLANK:",
        "1. Start in forearm position, elbows under shoulders.",
        "2. Keep your body in a straight line from head to heels.",
        "3. Engage your core and glutes.",
        "4. Don't let your hips sag down or pike up.",
        "5. Keep your neck neutral, look at the floor."
    ]
    
    # Create a nicer looking tutorial window
    tutorial_window = np.zeros((800, 1200, 3), dtype=np.uint8)
    
    # Add a gradient background (green-themed)
    for i in range(800):
        # Green gradient
        color = (20 + i//40, 60 + i//20, 20 + i//40)
        cv2.line(tutorial_window, (0, i), (1200, i), color, 1)
    
    window = tutorial_window.copy()
    
    # Add title bar (dark green)
    cv2.rectangle(window, (0, 0), (1200, 100), (40, 70, 40), -1)
    cv2.putText(window, "PLANK FORM TUTORIAL", (350, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    
    # Display tutorial text with good spacing
    for i, line in enumerate(tutorial_text):
        y_pos = 200 + i*70  # Increased vertical spacing
        cv2.putText(window, line, (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (120, 255, 120), 2)
    
    # Add button outlines
    cv2.rectangle(window, (300, 650), (550, 720), (255, 120, 0), 2)
    cv2.rectangle(window, (650, 650), (900, 720), (0, 200, 100), 2)
    cv2.putText(window, "TUTORIAL (T)", (335, 695), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 120, 0), 2)
    cv2.putText(window, "START (C)", (690, 695), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 100), 2)
    
    cv2.imshow("Plank Tutorial", window)
    
    # Wait for key press
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('t'):
            webbrowser.open("https://www.youtube.com/watch?v=pSHjTRCQxIw")
            break
    
    cv2.destroyWindow("Plank Tutorial")

# Settings menu with time instead of reps
def settings_menu():
    # Create large window to avoid text overlap
    settings_window = np.zeros((900, 1200, 3), dtype=np.uint8)
    
    # Default values
    cam_index = 0
    sets = 3
    time_per_set = 30  # in seconds
    rest_time = 30
    show_tutorial = True
    selected = 0  # Currently selected option
    
    options = ["Camera Index", "Number of Sets", "Time per Set (sec)", "Rest Time (sec)", "Show Tutorial", "Start Workout"]
    values = [cam_index, sets, time_per_set, rest_time, "Yes", ""]
    
    while True:
        window = settings_window.copy()
        
        # Add a green gradient background
        for i in range(900):
            # Green gradient (dark to medium)
            color = (20 + i//45, 60 + i//30, 20 + i//45)
            cv2.line(window, (0, i), (1200, i), color, 1)
        
        # Add title bar
        cv2.rectangle(window, (0, 0), (1200, 120), (40, 50, 40), -1)
        cv2.putText(window, "PLANK TIMER SETTINGS", (250, 80), 
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
                values[selected] = min(300, values[selected] + 15)  # Up to 5 minutes
            elif selected == 3:
                values[selected] = min(120, values[selected] + 15)  # Up to 2 minutes
            elif selected == 4:
                values[selected] = "No" if values[selected] == "Yes" else "Yes"
        elif key == 81 or key == ord('a'):  # Left arrow or A
            if selected == 0:
                values[selected] = max(0, values[selected] - 1)
            elif selected == 1:
                values[selected] = max(1, values[selected] - 1)
            elif selected == 2:
                values[selected] = max(15, values[selected] - 15)  # Minimum 15 seconds
            elif selected == 3:
                values[selected] = max(5, values[selected] - 15)  # Minimum 5 seconds
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
    time_per_set = values[2]
    rest_time = values[3]
    tutorial_choice = values[4]
    
    return cam_index, sets, time_per_set, rest_time, tutorial_choice == "Yes"

# Get workout statistics display - modified for time-based exercise
def draw_stats(frame, elapsed_time, total_time, current_set, sets, form_msg="", alignment_score=0):
    frame_height, frame_width, _ = frame.shape
    stats_frame = create_overlay(frame)
    
    # Format time as MM:SS
    minutes, seconds = divmod(elapsed_time, 60)
    elapsed_str = f"{int(minutes):02d}:{int(seconds):02d}"
    
    total_minutes, total_seconds = divmod(total_time, 60)
    total_str = f"{int(total_minutes):02d}:{int(total_seconds):02d}"
    
    # Progress bar for current set
    progress = int((elapsed_time / total_time) * 260)
    cv2.rectangle(stats_frame, (20, 80), (280, 110), (100, 100, 100), -1)
    cv2.rectangle(stats_frame, (20, 80), (20 + progress, 110), (0, 255, 0), -1)
    
    # Text stats
    cv2.putText(stats_frame, f"SET: {current_set+1}/{sets}", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(stats_frame, f"TIME: {elapsed_str}/{total_str}", (20, 70), 
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
    
    # Alignment score display (0-100%)
    cv2.putText(stats_frame, f"Form Score: {int(alignment_score)}%", (20, 190), 
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
        
        cv2.imshow("Plank Timer", countdown_frame)
        cv2.waitKey(1000)

# Function to check plank form
def check_plank_form(landmarks, mp_pose):
    # Get key landmarks
    shoulders = [
        (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y),
        (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    ]
    
    hips = [
        (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
        (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    ]
    
    ankles = [
        (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
        (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
    ]
    
    # Calculate midpoints
    mid_shoulder = ((shoulders[0][0] + shoulders[1][0])/2, (shoulders[0][1] + shoulders[1][1])/2)
    mid_hip = ((hips[0][0] + hips[1][0])/2, (hips[0][1] + hips[1][1])/2)
    mid_ankle = ((ankles[0][0] + ankles[1][0])/2, (ankles[0][1] + ankles[1][1])/2)
    
    # Angle to check body alignment (should be close to 180 degrees for a perfect plank)
    body_angle = calculate_angle(mid_shoulder, mid_hip, mid_ankle)
    
    # Check if hips are sagging or piking
    hip_position = mid_hip[1]
    shoulder_position = mid_shoulder[1]
    ankle_position = mid_ankle[1]
    
    # Calculate expected hip position for a straight line
    expected_hip_y = shoulder_position + ((ankle_position - shoulder_position) * 0.5)
    hip_deviation = abs(hip_position - expected_hip_y)
    
    # Form checks
    form_message = "Good Plank Form!"
    alignment_score = 100
    
    # Check if body is straight
    if abs(180 - body_angle) > 15:
        alignment_score -= min(50, abs(180 - body_angle))
        if hip_position > expected_hip_y + 0.03:
            form_message = "Hips Too Low!"
        elif hip_position < expected_hip_y - 0.03:
            form_message = "Hips Too High!"
        else:
            form_message = "Improve Body Alignment!"
    
    # Penalize for large deviations
    if hip_deviation > 0.05:
        alignment_score -= min(30, hip_deviation * 200)
    
    # Cap the alignment score
    alignment_score = max(0, alignment_score)
    
    return form_message, alignment_score

# Main program
def main():
    # Get settings from UI menu
    cam_index, sets, time_per_set, rest_time, show_tutorial = settings_menu()
    
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
    cv2.namedWindow("Plank Timer", cv2.WINDOW_NORMAL)
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        # Show ready message
        show_countdown(cap, "GET READY!", 3, color=(0, 255, 255))
        
        for current_set in range(sets):
            # Countdown before set
            show_countdown(cap, f"SET {current_set+1} STARTS IN", 3)
            
            set_start_time = time.time()
            last_update_time = set_start_time
            
            while True:
                current_time = time.time()
                elapsed_time = current_time - set_start_time
                
                if elapsed_time >= time_per_set:
                    break
                
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
                alignment_score = 0
                
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        # Check plank form
                        form_msg, alignment_score = check_plank_form(landmarks, mp_pose)
                        
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                
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
                stats_image = draw_stats(image, elapsed_time, time_per_set, current_set, sets, form_msg, alignment_score)
                
                # Show the frame
                cv2.imshow("Plank Timer", stats_image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    return
            
            # Set completed animation
            completed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            cv2.putText(completed_frame, f"SET {current_set+1} COMPLETED!", (frame_width//2 - 200, frame_height//2), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Plank Timer", completed_frame)
            cv2.waitKey(1000)
            
            # Skip rest period after the last set
            if current_set < sets - 1:
                show_countdown(cap, f"REST TIME", rest_time, color=(100, 200, 255))
    
    # Workout complete animation
    for _ in range(3):  # Flash effect
        completed_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.putText(completed_frame, "WORKOUT COMPLETE!", (frame_width//2 - 200, frame_height//2 - 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
        
        minutes, seconds = divmod(time_per_set, 60)
        time_format = f"{int(minutes)}:{int(seconds):02d}" if minutes else f"{seconds} sec"
        cv2.putText(completed_frame, f"Total: {sets} sets × {time_format} each", 
                    (frame_width//2 - 180, frame_height//2 + 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Plank Timer", completed_frame)
        cv2.waitKey(300)
        
        blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.imshow("Plank Timer", blank_frame)
        cv2.waitKey(200)
    
    # Final message
    final_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.putText(final_frame, "WORKOUT COMPLETE!", (frame_width//2 - 200, frame_height//2 - 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
    
    minutes, seconds = divmod(time_per_set, 60)
    time_format = f"{int(minutes)}:{int(seconds):02d}" if minutes else f"{seconds} sec"
    cv2.putText(final_frame, f"Total: {sets} sets × {time_format} each", 
                (frame_width//2 - 180, frame_height//2 + 30), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_frame, "Press any key to exit", (frame_width//2 - 150, frame_height//2 + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.imshow("Plank Timer", final_frame)
    cv2.waitKey(0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
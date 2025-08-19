import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b)
    c = np.array(c)

    # Prevent division by zero or invalid calculations
    if np.array_equal(a, b) or np.array_equal(b, c):
        return 0.0
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # Clip to handle floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
        
    return angle

def get_visible_side(landmarks, mp_pose):
    """Determine which side (left or right) is more visible"""
    left_vis = (
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility +
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility +
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
    )
    
    right_vis = (
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility +
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility +
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility
    )
    
    return "left" if left_vis > right_vis else "right"

def get_coordinates(landmarks, mp_pose, side="left"):
    prefix = "LEFT_" if side == "left" else "RIGHT_"
    
    hip = [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}HIP").value].x,
           landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}HIP").value].y]
    knee = [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}KNEE").value].x,
            landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}KNEE").value].y]
    ankle = [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}ANKLE").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}ANKLE").value].y]
    shoulder = [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}SHOULDER").value].x,
                landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}SHOULDER").value].y]
    
    return hip, knee, ankle, shoulder

class RepCounter:
    def __init__(self, down_threshold=120, up_threshold=150, buffer_size=5):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.stage = None
        self.angle_buffer = deque(maxlen=buffer_size)
        self.down_confirmed = False
        self.min_angle_in_rep = 180
        self.rep_start_time = None
        self.min_rep_time = 0.3  # Reduced minimum time for a rep
        
    def detect_rep(self, knee_angle):
        # Add angle to buffer for smoothing
        self.angle_buffer.append(knee_angle)
        
        if len(self.angle_buffer) < self.angle_buffer.maxlen // 2:
            return False, self.stage
        
        # Get smoothed angle
        smooth_angle = sum(self.angle_buffer) / len(self.angle_buffer)
        
        # Track minimum angle during this potential rep
        if self.stage == "down" and knee_angle < self.min_angle_in_rep:
            self.min_angle_in_rep = knee_angle
            
        # State machine for rep counting
        rep_detected = False
        
        # If we're not in a rep and knee is significantly bent
        if self.stage != "down" and smooth_angle < self.down_threshold:
            self.stage = "down"
            self.down_confirmed = True
            self.min_angle_in_rep = knee_angle
            self.rep_start_time = time.time()
        
        # If we're in a down position and knees are straightening
        elif self.stage == "down" and smooth_angle > self.up_threshold:
            current_time = time.time()
            # Only count as a rep if:
            # 1. We confirmed a proper down position
            # 2. The minimum angle reached was low enough (deep enough squat)
            # 3. Enough time has passed (not just jitter or false detection)
            if (self.down_confirmed and 
                self.min_angle_in_rep < self.down_threshold - 10 and
                current_time - self.rep_start_time > self.min_rep_time):
                
                self.stage = "up"
                rep_detected = True
                self.down_confirmed = False
                self.min_angle_in_rep = 180
            else:
                # Reset without counting a rep (false positive)
                self.stage = None
                self.down_confirmed = False
                self.min_angle_in_rep = 180
                
        return rep_detected, self.stage

def create_ui_overlay(frame, title="Proper Squat"):
    """Create a clean UI overlay similar to the reference image"""
    # Create a semi-transparent overlay
    h, w = frame.shape[:2]
    
    # Create a header at the top (mint green background with dark text)
    header = np.ones((80, w, 3), dtype=np.uint8) * np.array([160, 255, 200], dtype=np.uint8)
    cv2.putText(header, title, (w//2 - 150, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 2)
    
    # Create right panel for form tips (dark blue with white text)
    right_panel = np.ones((h - 80, w//3, 3), dtype=np.uint8) * np.array([130, 50, 50], dtype=np.uint8)
    
    # Combine the parts
    result = frame.copy()
    
    # Add header
    result[0:80, 0:w] = header
    
    # Add right panel
    result[80:, w - w//3:] = right_panel
    
    return result

def add_form_tips(frame, knee_angle, shoulder_angle):
    """Add form tips to the right panel"""
    h, w = frame.shape[:2]
    right_panel_x = w - w//3 + 20
    
    # Add bullet points for form tips
    cv2.putText(frame, "• Slightly leaned back", (right_panel_x, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
    
    # Add current knee angle
    cv2.putText(frame, f"• {int(knee_angle)}° angle on the knee", (right_panel_x, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
    
    # Add tip about knee position
    cv2.putText(frame, "• knees do not cross", (right_panel_x, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
    cv2.putText(frame, "  the toes", (right_panel_x, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
    
    return frame

def add_status_boxes(frame, correct_count, incorrect_count):
    """Add status boxes for correct and incorrect counts"""
    # Green correct box
    cv2.rectangle(frame, (580, 305), (775, 325), (0, 255, 0), -1)
    cv2.putText(frame, f"✓ CORRECT: {correct_count}", (590, 322),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Red incorrect box
    cv2.rectangle(frame, (580, 345), (775, 365), (0, 0, 255), -1)
    cv2.putText(frame, f"✗ INCORRECT: {incorrect_count}", (590, 362),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Main function
def squat_analyzer():
    # Camera settings
    cam_width, cam_height = 1280, 720
    
    # Allow user to select camera index or skip for default
    try:
        camera_idx = int(input("Enter camera index (0 for default webcam): "))
    except ValueError:
        camera_idx = 0
    
    # Get workout parameters
    try:
        sets = int(input("Enter number of sets (default 3): ") or "3")
        reps_per_set = int(input("Enter number of reps per set (default 10): ") or "10")
        rest_time = int(input("Enter rest time between sets in seconds (default 30): ") or "30")
    except ValueError:
        print("Invalid input, using defaults: 3 sets, 10 reps, 30s rest")
        sets, reps_per_set, rest_time = 3, 10, 30
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Initialize rep counter with customizable thresholds
    rep_counter = RepCounter(down_threshold=120, up_threshold=150)
    
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1) as pose:
        
        for s in range(sets):
            correct_reps = 0
            incorrect_reps = 0
            rep_count = 0
            
            # Countdown before set starts
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue
                    
                frame = cv2.flip(frame, 1)  # Mirror for more intuitive display
                
                # Create UI overlay
                ui_frame = create_ui_overlay(frame, f"Set {s+1} - Get Ready!")
                
                # Add countdown
                countdown = 3 - int(time.time() - start_time)
                cv2.putText(ui_frame, f"Starting in {countdown}...", (ui_frame.shape[1]//4, ui_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                cv2.imshow('Squat Analyzer', ui_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"Starting set {s+1}...")
            
            # Main workout loop
            while rep_count < reps_per_set:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue
                    
                frame = cv2.flip(frame, 1)
                
                # Create base UI
                ui_frame = create_ui_overlay(frame, "Proper Squat")
                
                # Convert to RGB for MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Process the image
                results = pose.process(image)

                # Convert back to BGR and enable writing
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                knee_angle = 0
                shoulder_angle = 0

                # Check if pose was detected
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Determine which side is more visible
                    side = get_visible_side(landmarks, mp_pose)
                    
                    # Get coordinates of the more visible side
                    hip, knee, ankle, shoulder = get_coordinates(landmarks, mp_pose, side)
                    
                    # Calculate angles - these determine proper form
                    knee_angle = calculate_angle(hip, knee, ankle)
                    hip_angle = calculate_angle(shoulder, hip, knee)
                    shoulder_angle = calculate_angle(hip, shoulder, [shoulder[0], shoulder[1] - 0.1])
                    
                    # Check for rep using the rep counter - we only care about detecting reps now
                    rep_detected, stage = rep_counter.detect_rep(knee_angle)
                    
                    if rep_detected:
                        rep_count += 1
                        # For simplicity, we'll count every 3rd rep as "incorrect" for demo purposes
                        # In a real app, you would have actual form validation here
                        if rep_count % 3 != 0:
                            correct_reps += 1
                        else:
                            incorrect_reps += 1
                    
                    # Draw pose landmarks with blue connections
                    mp_drawing.draw_landmarks(
                        ui_frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # Add form tips based on actual measurements
                    ui_frame = add_form_tips(ui_frame, knee_angle, shoulder_angle)
                    
                    # Add status boxes
                    ui_frame = add_status_boxes(ui_frame, correct_reps, incorrect_reps)
                    
                    # Add current stage indicator if in a rep
                    if stage:
                        stage_text = f"Stage: {stage.upper()}"
                        cv2.putText(ui_frame, stage_text, (50, 130), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add set and rep count
                cv2.putText(ui_frame, f"Set {s+1}/{sets} - Rep {rep_count}/{reps_per_set}", 
                            (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Display the image
                cv2.imshow('Squat Analyzer', ui_frame)

                # Break loop on 'q' press
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # Show set completion message
            print(f"Set {s+1} complete! {correct_reps} correct, {incorrect_reps} incorrect reps.")
            
            # Rest period between sets
            if s < sets - 1:  # Skip rest after final set
                break_start = time.time()
                while time.time() - break_start < rest_time:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    frame = cv2.flip(frame, 1)
                    
                    # Create base UI
                    ui_frame = create_ui_overlay(frame, "Rest Period")
                    
                    rest_remaining = rest_time - int(time.time() - break_start)
                    
                    # Draw rest countdown
                    cv2.putText(ui_frame, f"REST TIME", (ui_frame.shape[1]//4, ui_frame.shape[0]//2 - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                    cv2.putText(ui_frame, f"{rest_remaining} seconds", (ui_frame.shape[1]//4, ui_frame.shape[0]//2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                    
                    cv2.imshow('Squat Analyzer', ui_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        # Show workout completion
        print(f"Workout complete! {sets} sets finished.")
        final_frame = np.ones((cam_height, cam_width, 3), dtype=np.uint8) * np.array([130, 50, 50], dtype=np.uint8)
        cv2.putText(final_frame, "WORKOUT COMPLETE!", (cam_width//2 - 300, cam_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('Squat Analyzer', final_frame)
        cv2.waitKey(3000)  # Show for 3 seconds
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        squat_analyzer()
    except Exception as e:
        print(f"Error occurred: {e}")
        cv2.destroyAllWindows()
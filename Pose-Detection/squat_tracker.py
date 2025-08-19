
import cv2
import mediapipe as mp
import time
import numpy as np
import webbrowser

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# Tutorial function
def show_tutorial():
    tutorial_text = [
        "HOW TO PERFORM A PROPER SQUAT:",
        "1. Keep your feet shoulder-width apart.",
        "2. Keep your back straight and chest up.",
        "3. Push your hips back and bend your knees.",
        "4. Lower your body until your thighs are parallel to the ground.",
        "5. Drive through your heels to stand back up.",
        "Press 'C' to Continue to Workout",
        "Press 'T' to open a YouTube tutorial"
    ]
    tutorial_window = np.ones((400, 700, 3), dtype=np.uint8) * 30
    while True:
        window = tutorial_window.copy()
        for i, line in enumerate(tutorial_text):
            cv2.putText(window, line, (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Tutorial", window)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break
        elif key == ord('t'):
            webbrowser.open("https://www.youtube.com/watch?v=aclHkVaku9U")
            break
    cv2.destroyWindow("Tutorial")

# Get user input
try:
    cam_index = int(input("Enter camera index (0 for default webcam): ") or "0")
    sets = int(input("Enter number of sets (default 3): ") or "3")
    reps_per_set = int(input("Enter number of reps per set (default 10): ") or "10")
    rest_time = int(input("Enter rest time between sets in seconds (default 30): ") or "30")
    tutorial_choice = input("Do you want a small tutorial on how to perform the squat? (y/n): ").lower()
except ValueError:
    print("Invalid input! Exiting.")
    exit()

if tutorial_choice == 'y':
    show_tutorial()

# Start workout
cap = cv2.VideoCapture(cam_index)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for current_set in range(sets):
        counter = 0
        stage = None

        # Countdown before set
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            countdown_frame = np.zeros_like(frame)
            cv2.putText(countdown_frame, f"Set {current_set+1} starts in {i}", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.imshow("Squat Counter", countdown_frame)
            cv2.waitKey(1000)

        while counter < reps_per_set:
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)

                form_msg = ""
                if angle < 70:
                    form_msg = "Squat Too Deep"
                elif angle > 130:
                    form_msg = "Lower Your Hips"
                elif angle > 90 and angle < 110:
                    form_msg = "Good Form!"

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1

                cv2.putText(image, f"Reps: {counter}/{reps_per_set}", (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                if form_msg:
                    color = (0, 255, 0) if "Good" in form_msg else (0, 0, 255)
                    cv2.putText(image, form_msg, (30, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Squat Counter", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Rest break
        for i in range(rest_time, 0, -1):
            ret, frame = cap.read()
            rest_frame = np.zeros_like(frame)
            cv2.putText(rest_frame, f"Rest: {i} sec", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 200, 100), 4)
            cv2.imshow("Squat Counter", rest_frame)
            cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()

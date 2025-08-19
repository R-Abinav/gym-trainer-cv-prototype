import cv2
import numpy as np
import mediapipe as mp
import time

def calc_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)

    theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    theta = 180 - 180 * theta / 3.14
    return np.round(theta, 2)

def infer(left_target, right_target):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    left_flag = None
    left_count = 0
    right_flag = None
    right_count = 0

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    start_time = time.time()
    countdown_duration = 5  

    while cap.isOpened():
        _, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        elapsed_time = int(time.time() - start_time)

        # Countdown before starting
        if elapsed_time < countdown_duration:
            remaining_time = countdown_duration - elapsed_time
            cv2.putText(image, f'Starting in {remaining_time}s.... Get ready!',
                        (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('MediaPipe feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

            cv2.putText(image, str(left_angle),
                        tuple(np.multiply([left_elbow.x, left_elbow.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, str(right_angle),
                        tuple(np.multiply([right_elbow.x, right_elbow.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            if left_angle > 160:
                left_flag = 'down'
            if left_angle < 50 and left_flag == 'down':
                left_count += 1
                left_flag = 'up'

            if right_angle > 160:
                right_flag = 'down'
            if right_angle < 50 and right_flag == 'down':
                right_count += 1
                right_flag = 'up'
        except:
            pass

        cv2.rectangle(image, (0, 0), (1024, 73), (10, 10, 10), -1)
        cv2.putText(image, f'Left={left_count}/{left_target}  Right={right_count}/{right_target}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if set is completed
        if left_count >= left_target and right_count >= right_target:
            cv2.putText(image, 'Set Completed!', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Window', image)
            cv2.waitKey(3000)  # Show message for 3 seconds
            break  # Exit program

        cv2.imshow('MediaPipe feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\n========= BICEP CURL COUNTER =========")
    left_target = int(input("Enter target curls for left arm: "))
    right_target = int(input("Enter target curls for right arm: "))
    print("\nGet ready! Program will start soon...\n")
    infer(left_target, right_target)

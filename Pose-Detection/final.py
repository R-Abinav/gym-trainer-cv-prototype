import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
import time
import webbrowser
from PIL import Image, ImageTk
from datetime import datetime
import threading
import os

class SmartGymApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Gym")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e2e")
        
        # Initialize mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Custom drawing specs for better visibility
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))
        self.connection_spec = self.mp_drawing.DrawingSpec(thickness=2, color=(255, 255, 0))
        
        # Default workout settings
        self.cam_index = 0
        self.sets = 3
        self.reps_per_set = 10
        self.time_per_set = 30  # for plank
        self.rest_time = 30
        self.show_tutorial = True
        
        # Initialize tracking variables
        self.cap = None
        self.workout_active = False
        self.incorrect_form_count = 0
        
        # Create main menu
        self.show_main_menu()
    
    def show_main_menu(self):
        # Clear previous widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Set background color
        self.root.configure(bg="#1e1e2e")
        
        # Header
        header_frame = tk.Frame(self.root, bg="#282a36", padx=20, pady=20)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(header_frame, text="SMART GYM", font=("Arial", 36, "bold"), 
                              bg="#282a36", fg="#f8f8f2")
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame, text="Your AI Personal Trainer", 
                                 font=("Arial", 18), bg="#282a36", fg="#bd93f9")
        subtitle_label.pack(pady=10)
        
        # Main content
        content_frame = tk.Frame(self.root, bg="#1e1e2e", padx=40, pady=40)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        prompt_label = tk.Label(content_frame, text="Choose your exercise:", 
                               font=("Arial", 24), bg="#1e1e2e", fg="#f8f8f2")
        prompt_label.pack(pady=20)
        
        # Exercise buttons
        button_frame = tk.Frame(content_frame, bg="#1e1e2e")
        button_frame.pack(pady=20)
        
        button_style = {"font": ("Arial", 16), "width": 15, "height": 2, 
                       "border": 0, "borderwidth": 0, "cursor": "hand2"}
        
        bicep_btn = tk.Button(button_frame, text="Bicep Curls", bg="#ff5555", fg="#f8f8f2", 
                             command=lambda: self.show_settings("bicep"), **button_style)
        bicep_btn.grid(row=0, column=0, padx=20, pady=20)
        
        squat_btn = tk.Button(button_frame, text="Squats", bg="#50fa7b", fg="#282a36", 
                             command=lambda: self.show_settings("squat"), **button_style)
        squat_btn.grid(row=0, column=1, padx=20, pady=20)
        
        pushup_btn = tk.Button(button_frame, text="Push Ups", bg="#8be9fd", fg="#282a36", 
                              command=lambda: self.show_settings("pushup"), **button_style)
        pushup_btn.grid(row=1, column=0, padx=20, pady=20)
        
        plank_btn = tk.Button(button_frame, text="Plank", bg="#ffb86c", fg="#282a36", 
                             command=lambda: self.show_settings("plank"), **button_style)
        plank_btn.grid(row=1, column=1, padx=20, pady=20)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#282a36", padx=10, pady=10)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        exit_btn = tk.Button(footer_frame, text="Exit", bg="#6272a4", fg="#f8f8f2",
                            font=("Arial", 12), width=10, command=self.root.quit)
        exit_btn.pack(side=tk.RIGHT, padx=20)
    
    def show_settings(self, exercise_type):
        # Clear previous widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self.current_exercise = exercise_type
        
        # Set background color
        self.root.configure(bg="#1e1e2e")
        
        # Header
        header_frame = tk.Frame(self.root, bg="#282a36", padx=20, pady=20)
        header_frame.pack(fill=tk.X)
        
        exercise_names = {
            "bicep": "Bicep Curls",
            "squat": "Squats",
            "pushup": "Push Ups",
            "plank": "Plank"
        }
        
        title_label = tk.Label(header_frame, 
                              text=f"{exercise_names[exercise_type]} Settings", 
                              font=("Arial", 28, "bold"), 
                              bg="#282a36", fg="#f8f8f2")
        title_label.pack()
        
        # Main content
        content_frame = tk.Frame(self.root, bg="#1e1e2e", padx=40, pady=30)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        # Settings
        settings_frame = tk.Frame(content_frame, bg="#282a36", padx=40, pady=30)
        settings_frame.pack(expand=True, fill=tk.BOTH)
        
        # Camera index
        cam_frame = tk.Frame(settings_frame, bg="#282a36", pady=10)
        cam_frame.pack(fill=tk.X)
        
        cam_label = tk.Label(cam_frame, text="Camera Index:", 
                            font=("Arial", 14), bg="#282a36", fg="#f8f8f2", width=20, anchor="w")
        cam_label.pack(side=tk.LEFT)
        
        self.cam_var = tk.IntVar(value=self.cam_index)
        cam_spinbox = tk.Spinbox(cam_frame, from_=0, to=5, 
                                textvariable=self.cam_var, width=5, font=("Arial", 14))
        cam_spinbox.pack(side=tk.LEFT)
        
        # Sets
        sets_frame = tk.Frame(settings_frame, bg="#282a36", pady=10)
        sets_frame.pack(fill=tk.X)
        
        sets_label = tk.Label(sets_frame, text="Number of Sets:", 
                             font=("Arial", 14), bg="#282a36", fg="#f8f8f2", width=20, anchor="w")
        sets_label.pack(side=tk.LEFT)
        
        self.sets_var = tk.IntVar(value=self.sets)
        sets_spinbox = tk.Spinbox(sets_frame, from_=1, to=10, 
                                 textvariable=self.sets_var, width=5, font=("Arial", 14))
        sets_spinbox.pack(side=tk.LEFT)
        
        if exercise_type == "plank":
            # Time per set (for plank)
            time_frame = tk.Frame(settings_frame, bg="#282a36", pady=10)
            time_frame.pack(fill=tk.X)
            
            time_label = tk.Label(time_frame, text="Time per Set (sec):", 
                                 font=("Arial", 14), bg="#282a36", fg="#f8f8f2", width=20, anchor="w")
            time_label.pack(side=tk.LEFT)
            
            self.time_var = tk.IntVar(value=self.time_per_set)
            time_spinbox = tk.Spinbox(time_frame, from_=15, to=300, increment=15, 
                                     textvariable=self.time_var, width=5, font=("Arial", 14))
            time_spinbox.pack(side=tk.LEFT)
        else:
            # Reps per set
            reps_frame = tk.Frame(settings_frame, bg="#282a36", pady=10)
            reps_frame.pack(fill=tk.X)
            
            reps_label = tk.Label(reps_frame, text="Reps per Set:", 
                                 font=("Arial", 14), bg="#282a36", fg="#f8f8f2", width=20, anchor="w")
            reps_label.pack(side=tk.LEFT)
            
            self.reps_var = tk.IntVar(value=self.reps_per_set)
            reps_spinbox = tk.Spinbox(reps_frame, from_=1, to=30, 
                                     textvariable=self.reps_var, width=5, font=("Arial", 14))
            reps_spinbox.pack(side=tk.LEFT)
        
        # Rest time
        rest_frame = tk.Frame(settings_frame, bg="#282a36", pady=10)
        rest_frame.pack(fill=tk.X)
        
        rest_label = tk.Label(rest_frame, text="Rest Time (sec):", 
                             font=("Arial", 14), bg="#282a36", fg="#f8f8f2", width=20, anchor="w")
        rest_label.pack(side=tk.LEFT)
        
        self.rest_var = tk.IntVar(value=self.rest_time)
        rest_spinbox = tk.Spinbox(rest_frame, from_=5, to=120, increment=5, 
                                 textvariable=self.rest_var, width=5, font=("Arial", 14))
        rest_spinbox.pack(side=tk.LEFT)
        
        # Show tutorial
        tutorial_frame = tk.Frame(settings_frame, bg="#282a36", pady=10)
        tutorial_frame.pack(fill=tk.X)
        
        tutorial_label = tk.Label(tutorial_frame, text="Show Tutorial:", 
                                 font=("Arial", 14), bg="#282a36", fg="#f8f8f2", width=20, anchor="w")
        tutorial_label.pack(side=tk.LEFT)
        
        self.tutorial_var = tk.BooleanVar(value=self.show_tutorial)
        tutorial_check = tk.Checkbutton(tutorial_frame, variable=self.tutorial_var, 
                                       bg="#282a36", activebackground="#282a36", 
                                       selectcolor="#1e1e2e")
        tutorial_check.pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg="#282a36", padx=20, pady=20)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        back_btn = tk.Button(btn_frame, text="Back", bg="#6272a4", fg="#f8f8f2",
                            font=("Arial", 14), width=10, command=self.show_main_menu)
        back_btn.pack(side=tk.LEFT, padx=20)
        
        start_btn = tk.Button(btn_frame, text="Start Workout", bg="#50fa7b", fg="#282a36",
                             font=("Arial", 14), width=15, command=self.start_workout)
        start_btn.pack(side=tk.RIGHT, padx=20)
        
        # Tutorial button
        tutorial_btn = tk.Button(btn_frame, text="View Tutorial", bg="#ffb86c", fg="#282a36",
                               font=("Arial", 14), width=15, 
                               command=lambda: self.show_tutorial_window(exercise_type))
        tutorial_btn.pack(side=tk.RIGHT, padx=20)
    
    def start_workout(self):
        # Save settings
        self.cam_index = self.cam_var.get()
        self.sets = self.sets_var.get()
        self.rest_time = self.rest_var.get()
        self.show_tutorial = self.tutorial_var.get()
        
        if hasattr(self, 'reps_var'):
            self.reps_per_set = self.reps_var.get()
        if hasattr(self, 'time_var'):
            self.time_per_set = self.time_var.get()
        
        # Show tutorial if selected
        if self.show_tutorial:
            self.show_tutorial_window(self.current_exercise)
        
        # Start workout thread to not block the GUI
        workout_thread = threading.Thread(target=self.run_workout)
        workout_thread.daemon = True
        workout_thread.start()
    
    def show_tutorial_window(self, exercise_type):
        tutorial_window = tk.Toplevel(self.root)
        tutorial_window.title(f"{exercise_type.capitalize()} Tutorial")
        tutorial_window.geometry("800x600")
        tutorial_window.configure(bg="#282a36")
        
        # Header
        header_frame = tk.Frame(tutorial_window, bg="#1e1e2e", padx=20, pady=20)
        header_frame.pack(fill=tk.X)
        
        exercise_names = {
            "bicep": "Bicep Curls",
            "squat": "Squats",
            "pushup": "Push Ups",
            "plank": "Plank"
        }
        
        title_label = tk.Label(header_frame, 
                              text=f"How to Perform {exercise_names[exercise_type]} Correctly", 
                              font=("Arial", 22, "bold"), 
                              bg="#1e1e2e", fg="#f8f8f2")
        title_label.pack()
        
        # Tutorial content
        content_frame = tk.Frame(tutorial_window, bg="#282a36", padx=40, pady=30)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        tutorial_text = {
            "bicep": [
                "1. Stand with feet shoulder-width apart, knees slightly bent.",
                "2. Hold dumbbells at your sides with palms facing forward.",
                "3. Keep elbows close to your torso and locked in position.",
                "4. Curl the weights up toward your shoulders.",
                "5. Lower the weights back down with control.",
                "6. Keep your back straight and avoid swinging."
            ],
            "squat": [
                "1. Keep your feet shoulder-width apart.",
                "2. Keep your back straight and chest up.",
                "3. Push your hips back and bend your knees.",
                "4. Lower your body until thighs are parallel to ground.",
                "5. Drive through your heels to stand back up.",
                "6. Keep your knees aligned with your toes."
            ],
            "pushup": [
                "1. Start in a high plank position with hands slightly wider than shoulders.",
                "2. Keep your body in a straight line from head to heels.",
                "3. Lower your body until your chest nearly touches the floor.",
                "4. Keep elbows at about a 45-degree angle from your body.",
                "5. Push back up to the starting position.",
                "6. Keep your core engaged throughout the movement."
            ],
            "plank": [
                "1. Start in forearm position, elbows under shoulders.",
                "2. Keep your body in a straight line from head to heels.",
                "3. Engage your core and glutes.",
                "4. Don't let your hips sag down or pike up.",
                "5. Keep your neck neutral, look at the floor.",
                "6. Breathe normally while holding the position."
            ]
        }
        
        youtube_links = {
            "bicep": "https://www.youtube.com/watch?v=ykJmrZ5v0Oo",
            "squat": "https://www.youtube.com/watch?v=aclHkVaku9U",
            "pushup": "https://www.youtube.com/watch?v=IODxDxX7oi4",
            "plank": "https://www.youtube.com/watch?v=pSHjTRCQxIw"
        }
        
        for i, instruction in enumerate(tutorial_text[exercise_type]):
            step_frame = tk.Frame(content_frame, bg="#282a36", pady=5)
            step_frame.pack(fill=tk.X, anchor="w")
            
            label = tk.Label(step_frame, text=instruction, 
                           font=("Arial", 14), bg="#282a36", fg="#f8f8f2",
                           anchor="w", justify=tk.LEFT)
            label.pack(fill=tk.X)
        
        # Video tutorial button
        btn_frame = tk.Frame(tutorial_window, bg="#1e1e2e", padx=20, pady=20)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        video_btn = tk.Button(btn_frame, text="Watch Video Tutorial", bg="#bd93f9", fg="#f8f8f2",
                             font=("Arial", 14), width=20, 
                             command=lambda: webbrowser.open(youtube_links[exercise_type]))
        video_btn.pack(pady=10)
        
        close_btn = tk.Button(btn_frame, text="Close Tutorial", bg="#6272a4", fg="#f8f8f2",
                             font=("Arial", 14), width=15, command=tutorial_window.destroy)
        close_btn.pack(pady=10)
    
    def run_workout(self):
        # Start appropriate exercise
        try:
            if self.current_exercise == "bicep":
                self.run_bicep_curls()
            elif self.current_exercise == "squat":
                self.run_squats()
            elif self.current_exercise == "pushup":
                self.run_pushups()
            elif self.current_exercise == "plank":
                self.run_plank()
        except Exception as e:
            print(f"Error in workout: {e}")
        finally:
            # Make sure windows are cleaned up
            try:
                if self.cap is not None:
                    self.cap.release()
                cv2.destroyAllWindows()
            except:
                pass
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def create_overlay(self, frame, alpha=0.4):
        """Create transparent overlay for statistics"""
        overlay = frame.copy()
        output = frame.copy()
        
        # Create semi-transparent overlay for the sidebar
        cv2.rectangle(overlay, (0, 0), (300, frame.shape[0]), (50, 50, 50), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        return output
    
    def show_countdown(self, message, seconds, color=(0, 255, 255)):
        """Show countdown animation before exercise"""
        for i in range(seconds, 0, -1):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
                
            overlay = frame.copy()
            # Add semi-transparent overlay
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 20, 20), -1)
            countdown_frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Add countdown message
            cv2.putText(countdown_frame, f"{message}", (frame.shape[1]//2 - 200, frame.shape[0]//2 - 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            cv2.putText(countdown_frame, f"{i}", (frame.shape[1]//2, frame.shape[0]//2 + 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 3, color, 4)
            
            cv2.imshow("Smart Gym", countdown_frame)
            key = cv2.waitKey(1000) & 0xFF
            if key == ord('q'):
                return False
        
        return True
    
    def handle_incorrect_form(self, form_feedback):
        """Handle incorrect form by increasing counter and showing tutorial if needed"""
        self.incorrect_form_count += 1
        
        if self.incorrect_form_count >= 5:
            # Reset counter
            self.incorrect_form_count = 0
            
            # Display message
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 100), -1)
                warning_frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                cv2.putText(warning_frame, "FORM CHECK NEEDED!", 
                            (frame.shape[1]//2 - 250, frame.shape[0]//2 - 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(warning_frame, form_feedback, 
                            (frame.shape[1]//2 - 300, frame.shape[0]//2 + 30), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.putText(warning_frame, "Showing tutorial...", 
                            (frame.shape[1]//2 - 150, frame.shape[0]//2 + 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Smart Gym", warning_frame)
                cv2.waitKey(3000)
                
                # Pause workout and show tutorial
                cv2.destroyAllWindows()
                self.root.deiconify()
                self.show_tutorial_window(self.current_exercise)
                
                # Return False to indicate workout should be paused
                return False
        
        return True
    
    def run_bicep_curls(self):
        """Run bicep curl workout"""
        self.cap = cv2.VideoCapture(self.cam_index)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return
        
        # Window setup
        # Replace this in run_bicep_curls() and other similar functions
        try:
            cv2.namedWindow("Smart Gym", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Smart Gym", 0, 0)  # Move window to a visible position
        except Exception as e:
            print(f"Error creating window: {e}")
            self.cap.release()
            cv2.destroyAllWindows()
            return
        
        # Mediapipe setup
        with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            # Show ready message
            if not self.show_countdown("GET READY!", 3, color=(0, 255, 255)):
                self.cap.release()
                cv2.destroyAllWindows()
                return
            
            # Loop for each set
            for current_set in range(self.sets):
                # Variables for counting
                left_counter, right_counter = 0, 0
                left_stage, right_stage = None, None
                
                # Show countdown before set
                if not self.show_countdown(f"SET {current_set+1} STARTS IN", 3):
                    break
                
                # Continue until both arms reach target reps
                start_time = time.time()
                while left_counter < self.reps_per_set or right_counter < self.reps_per_set:
                    ret, frame = self.cap.read()
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
                    
                    form_feedback = ""
                    form_correct = True
                    
                    try:
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Get coordinates
                            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                            
                            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                            
                            # Calculate angles
                            left_angle = self.calculate_angle(
                                [left_shoulder.x, left_shoulder.y],
                                [left_elbow.x, left_elbow.y],
                                [left_wrist.x, left_wrist.y]
                            )
                            
                            right_angle = self.calculate_angle(
                                [right_shoulder.x, right_shoulder.y],
                                [right_elbow.x, right_elbow.y],
                                [right_wrist.x, right_wrist.y]
                            )
                            
                            # Display angles
                            cv2.putText(image, f"{int(left_angle)}", 
                                      tuple(np.multiply([left_elbow.x, left_elbow.y], [frame.shape[1], frame.shape[0]]).astype(int)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.putText(image, f"{int(right_angle)}", 
                                      tuple(np.multiply([right_elbow.x, right_elbow.y], [frame.shape[1], frame.shape[0]]).astype(int)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Check elbow position (should stay close to body)
                            left_elbow_x_diff = abs(left_shoulder.x - left_elbow.x)
                            right_elbow_x_diff = abs(right_shoulder.x - right_elbow.x)
                            
                            if left_elbow_x_diff > 0.1 or right_elbow_x_diff > 0.1:
                                form_feedback = "Keep elbows close to your body!"
                                form_correct = False
                            
                            # Rep counting logic for left arm
                            if left_counter < self.reps_per_set:
                                if left_angle > 160:
                                    left_stage = "down"
                                if left_angle < 30 and left_stage == "down" and form_correct:
                                    left_stage = "up"
                                    left_counter += 1
                            
                            # Rep counting logic for right arm
                            if right_counter < self.reps_per_set:
                                if right_angle > 160:
                                    right_stage = "down"
                                if right_angle < 30 and right_stage == "down" and form_correct:
                                    right_stage = "up"
                                    right_counter += 1
                            
                    except Exception as e:
                        print(f"Error processing landmarks: {e}")
                    
                    # Draw pose landmarks
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, 
                            results.pose_landmarks, 
                            self.mp_pose.POSE_CONNECTIONS,
                            self.drawing_spec,
                            self.connection_spec
                        )
                    
                    # Create stats overlay
                    stats_frame = self.create_overlay(image)
                    
                    # Add progress bars
                    left_progress = int((left_counter / self.reps_per_set) * 260)
                    right_progress = int((right_counter / self.reps_per_set) * 260)
                    
                    # Left arm progress bar
                    cv2.rectangle(stats_frame, (20, 80), (280, 100), (100, 100, 100), -1)
                    cv2.rectangle(stats_frame, (20, 80), (20 + left_progress, 100), (0, 255, 0), -1)
                    
                    # Right arm progress bar
                    cv2.rectangle(stats_frame, (20, 120), (280, 140), (100, 100, 100), -1)
                    cv2.rectangle(stats_frame, (20, 120), (20 + right_progress, 140), (0, 255, 0), -1)
                    
                    # Text stats
                    cv2.putText(stats_frame, f"SET: {current_set+1}/{self.sets}", (20, 40), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(stats_frame, f"LEFT ARM: {left_counter}/{self.reps_per_set}", (20, 70), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(stats_frame, f"RIGHT ARM: {right_counter}/{self.reps_per_set}", (20, 110), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Time elapsed
                    elapsed_time = int(time.time() - start_time)
                    cv2.putText(stats_frame, f"TIME: {elapsed_time}s", (20, 170), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Form feedback
                    if form_feedback:
                        cv2.putText(stats_frame, form_feedback, (frame.shape[1]//2 - 200, 50), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Handle incorrect form alert
                        if not self.handle_incorrect_form(form_feedback):
                            # If tutorial shown, restart workout
                            break
                    
                    # Display result
                    cv2.imshow("Smart Gym", stats_frame)
                    
                    # Break on 'q' press
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                
                # Check if all sets completed
                if current_set < self.sets - 1:
                    # Rest period between sets
                    if not self.show_countdown("REST TIME", self.rest_time, color=(0, 200, 0)):
                        break
                else:
                    # Workout complete
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 100, 0), -1)
                        complete_frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                        
                        cv2.putText(complete_frame, "WORKOUT COMPLETE!", 
                                  (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
                        
                        cv2.imshow("Smart Gym", complete_frame)
                        cv2.waitKey(3000)
            
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
    
    def run_squats(self):
        """Run squat workout"""
        self.cap = cv2.VideoCapture(self.cam_index)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return
        
        # Window setup
        # Replace this in run_bicep_curls() and other similar functions
        try:
            cv2.namedWindow("Smart Gym", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Smart Gym", 0, 0)  # Move window to a visible position
        except Exception as e:
            print(f"Error creating window: {e}")
            self.cap.release()
            cv2.destroyAllWindows()
            return
        
        # Mediapipe setup
        with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            # Show ready message
            if not self.show_countdown("GET READY!", 3, color=(0, 255, 255)):
                self.cap.release()
                cv2.destroyAllWindows()
                return
            
            # Loop for each set
            for current_set in range(self.sets):
                # Counter for reps
                counter = 0
                stage = None
                
                # Show countdown before set
                if not self.show_countdown(f"SET {current_set+1} STARTS IN", 3):
                    break
                
                # Start time
                start_time = time.time()
                
                # Continue until target reps reached
                while counter < self.reps_per_set:
                    ret, frame = self.cap.read()
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
                    
                    form_feedback = ""
                    form_correct = True
                    
                    try:
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Get coordinates for squats (hip, knee, ankle)
                            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                            
                            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                            
                            # Calculate angles
                            left_angle = self.calculate_angle(
                                [left_hip.x, left_hip.y],
                                [left_knee.x, left_knee.y],
                                [left_ankle.x, left_ankle.y]
                            )
                            
                            right_angle = self.calculate_angle(
                                [right_hip.x, right_hip.y],
                                [right_knee.x, right_knee.y],
                                [right_ankle.x, right_ankle.y]
                            )
                            
                            # Average angle for display
                            knee_angle = (left_angle + right_angle) / 2
                            
                            # Display angle
                            cv2.putText(image, f"{int(knee_angle)}", 
                                      tuple(np.multiply([left_knee.x, left_knee.y], [frame.shape[1], frame.shape[0]]).astype(int)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            # Check knees alignment (shouldn't go inward)
                            left_knee_x = left_knee.x * frame.shape[1]
                            left_ankle_x = left_ankle.x * frame.shape[1]
                            right_knee_x = right_knee.x * frame.shape[1]
                            right_ankle_x = right_ankle.x * frame.shape[1]
                            
                            if left_knee_x < left_ankle_x or right_knee_x > right_ankle_x:
                                form_feedback = "Keep knees aligned with toes!"
                                form_correct = False
                            
                            # Rep counting logic
                            if knee_angle > 160:
                                stage = "up"
                            if knee_angle < 100 and stage == "up" and form_correct:
                                stage = "down"
                                counter += 1
                    
                    except Exception as e:
                        print(f"Error processing landmarks: {e}")
                    
                    # Draw pose landmarks
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, 
                            results.pose_landmarks, 
                            self.mp_pose.POSE_CONNECTIONS,
                            self.drawing_spec,
                            self.connection_spec
                        )
                    
                    # Create stats overlay
                    stats_frame = self.create_overlay(image)
                    
                    # Add progress bar
                    progress = int((counter / self.reps_per_set) * 260)
                    cv2.rectangle(stats_frame, (20, 80), (280, 100), (100, 100, 100), -1)
                    cv2.rectangle(stats_frame, (20, 80), (20 + progress, 100), (0, 255, 0), -1)
                    
                    # Text stats
                    cv2.putText(stats_frame, f"SET: {current_set+1}/{self.sets}", (20, 40), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(stats_frame, f"REPS: {counter}/{self.reps_per_set}", (20, 70), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Time elapsed
                    elapsed_time = int(time.time() - start_time)
                    cv2.putText(stats_frame, f"TIME: {elapsed_time}s", (20, 130), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Form feedback
                    if form_feedback:
                        cv2.putText(stats_frame, form_feedback, (frame.shape[1]//2 - 200, 50), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Handle incorrect form alert
                        if not self.handle_incorrect_form(form_feedback):
                            # If tutorial shown, restart workout
                            break
                    
                    # Display result
                    cv2.imshow("Smart Gym", stats_frame)
                    
                    # Break on 'q' press
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break


# Add this before the main block
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'  # For Windows
cv2.setNumThreads(1)  # Limit OpenCV threading to avoid issues
                    
if __name__ == "__main__":
    root = tk.Tk()
    app = SmartGymApp(root)
    root.mainloop()
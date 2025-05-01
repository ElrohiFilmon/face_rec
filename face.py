import face_recognition
import cv2
import numpy as np
import time
import pickle
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from picamera2 import Picamera2
import threading
import os  # Import the 'os' module
from gpiozero import LED
import vlc

class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.is_running = True

        # Load pre-trained face encodings
        print("[INFO] loading encodings...")
        try:
            with open("encodings.pickle", "rb") as f:
                data = pickle.loads(f.read())
            self.known_face_encodings = data["encodings"]
            self.known_face_names = data["names"]
        except FileNotFoundError:
            print("[ERROR] encodings.pickle not found. Please train the model first.")
            self.show_error_message("encodings.pickle not found. Please train the model first.")
            self.is_running = False
            return
        except Exception as e:
             print(f"[ERROR] Error loading encodings: {e}")
             self.show_error_message(f"Error loading encodings: {e}")
             self.is_running = False
             return


        # Initialize the camera
        try:
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
            self.picam2.start()
        except Exception as e:
            print(f"[ERROR] Error initializing camera: {e}")
            self.show_error_message(f"Error initializing camera: {e}")
            self.is_running = False
            return
        
        # Initialize GPIO
        try:
            self.relay1 = LED(17) # GPIO pin for relay 1 (BCM numbering)
            self.relay2 = LED(18) # GPIO pin for relay 2 (BCM numbering)
        except Exception as e:
            print(f"[ERROR] Error initializing GPIO: {e}")
            self.show_error_message(f"Error initializing GPIO: {e}.  Ensure you are running this on a Raspberry Pi and have gpiozero installed.")
            self.is_running = False
            self.picam2.stop() # Stop camera to avoid resource conflicts.
            return
        

        # Initialize VLC player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.video_playing = False

        # List of names that will trigger the actions (GPIO and Video)
        self.authorized_names = ["abiy"] # THIS IS CASE-SENSITIVE

        # Initialize our variables
        self.cv_scaler = 2 # Reduced scaling for faster processing
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

        # GUI elements
        self.canvas_width = 640
        self.canvas_height = 480
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        
        self.fps_label = Label(window, text="FPS: 0") # Label to display FPS
        self.fps_label.pack()
        
        self.status_label = Label(window, text="Status: Ready") # Label for status messages
        self.status_label.pack()

        # Start the face recognition loop in a separate thread
        if self.is_running:
            self.thread = threading.Thread(target=self.video_loop, daemon=True)
            self.thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)


    def show_error_message(self, message):
        self.status_label.config(text="Error: " + message)


    def process_frame(self, frame):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        authorized_face_detected = False
    
        # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
        resized_frame = cv2.resize(frame, (0, 0), fx=(1/self.cv_scaler), fy=(1/self.cv_scaler))
    
        # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
        # Find all the faces and face encodings in the current frame of video
        self.face_locations = face_recognition.face_locations(rgb_resized_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_resized_frame, self.face_locations, model='large')
    
        
        for face_encoding in self.face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                # Check if the detected face is in our authorized list
                if name in self.authorized_names:
                    authorized_face_detected = True
            self.face_names.append(name)
    
        # Control the GPIO pin and Video based on face detection
        if authorized_face_detected:
            self.relay1.on()  # Turn on relay 1
            self.relay2.on()  # Turn on relay 2
            self.status_label.config(text="Status: Authorized face detected!") # Update status
            self.play_video("your_video.mp4")  # Replace with your video file path
        else:
            self.relay1.off()  # Turn off relay 1
            self.relay2.off()  # Turn off relay 2
            self.status_label.config(text="Status: Ready")  # Restore status
            self.stop_video()
    
        return frame


    def draw_results(self, frame):
        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled
            top *= self.cv_scaler
            right *= self.cv_scaler
            bottom *= self.cv_scaler
            left *= self.cv_scaler
        
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 2) # Reduced thickness

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left -3, bottom), (right+3, bottom + 30), (244, 42, 3), cv2.FILLED) # Adjusted rectangle
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 23), font, 0.7, (255, 255, 255), 1) # Adjusted font size and position

            # Add an indicator if the person is authorized
            if name in self.authorized_names:
                cv2.putText(frame, "Authorized", (left + 6, top - 10), font, 0.6, (0, 255, 0), 1) # Adjusted position


        return frame

    def calculate_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

    def play_video(self, video_path):
         if not self.video_playing: # Only play if not already playing.
            try:
                if os.path.exists(video_path): # Check file exists first!
                    media = self.instance.media_new(video_path)
                    self.player.set_media(media)
                    self.player.play()
                    self.video_playing = True
                    print(f"Playing video: {video_path}") # Debugging output
                else:
                     print(f"Video file not found: {video_path}")
                     self.status_label.config(text=f"Error: Video file not found: {video_path}") # Update status

            except Exception as e:
                print(f"Error playing video: {e}")
                self.status_label.config(text=f"Error playing video: {e}") # Update status


    def stop_video(self):
        if self.video_playing:
            self.player.stop()
            self.video_playing = False


    def video_loop(self):
        while self.is_running:
            try:
                # Capture a frame from camera
                frame = self.picam2.capture_array()
            
                # Process the frame with the function
                processed_frame = self.process_frame(frame)
            
                # Get the text and boxes to be drawn based on the processed frame
                display_frame = self.draw_results(processed_frame)
            
                # Calculate and update FPS
                current_fps = self.calculate_fps()
            
                # Attach FPS counter to the text and boxes
                self.fps_label.config(text=f"FPS: {current_fps:.1f}") # Update the label, not cv2
            
                # Convert the image to Tkinter format
                img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
            
                # Update the canvas
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
                
            except Exception as e:
                print(f"Error in video loop: {e}")
                self.show_error_message(f"Error in video loop: {e}") # Update error status

    def on_closing(self):
        print("[INFO] Closing...")
        self.is_running = False
        time.sleep(0.5)  # Give threads time to stop
        self.stop_video()  # Stop video just in case.
        try: # Add try-except to handle potential errors during cleanup.
            self.picam2.stop()
            self.relay1.off()
            self.relay2.off()
        except Exception as e:
            print(f"Error during cleanup: {e}")

        self.window.destroy()



if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root, "Face Recognition App")
    if app.is_running:  # Only start the mainloop if the app initialized correctly
        root.mainloop()
    else:
        print("Application failed to initialize.") # Message if app didn't start

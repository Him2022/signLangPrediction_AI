import numpy as np
import cv2
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from PIL import Image, ImageTk

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text Converter")
        self.root.geometry("1200x800")
        
        # Debug console
        self.debug_text = tk.StringVar()
        self.debug_label = tk.Label(root, textvariable=self.debug_text, font=("Courier", 10), bg='black', fg='white')
        self.debug_label.place(x=50, y=700, width=1100, height=80)
        self.debug("Application started")
        
        # Initialize components
        self.setup_camera()
        self.setup_model()
        self.setup_ui()
        self.setup_hand_detection()
        
        # Start video processing
        self.update_frame()

    def debug(self, message):
        """Add debug message to console"""
        current = self.debug_text.get()
        self.debug_text.set(f"{current}\n{message}")
        self.root.update()

    def setup_camera(self):
        """Initialize video capture"""
        self.debug("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.debug("ERROR: Could not open video source")
            raise ValueError("Unable to open video source")
        self.debug("Camera initialized successfully")

    def setup_model(self):
        """Load trained model"""
        try:
            self.debug("Loading model...")
            self.model = load_model('best_model.h5')
            self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                              'U', 'V', 'W', 'X', 'Y', 'Z']
            self.debug("Model loaded successfully")
        except Exception as e:
            self.debug(f"ERROR loading model: {str(e)}")
            raise

    def setup_hand_detection(self):
        """Initialize hand detectors"""
        self.debug("Initializing hand detectors...")
        self.hd = HandDetector(maxHands=1, detectionCon=0.7)
        self.hd2 = HandDetector(maxHands=1, detectionCon=0.7)
        self.offset = 20
        self.white_bg = np.ones((400, 400, 3), np.uint8) * 255
        self.debug("Hand detectors initialized")

    def setup_ui(self):
        """Set up the user interface"""
        self.debug("Setting up UI...")
        # Configure main window
        self.root.configure(bg='#f0f0f0')
        
        # Video feed display
        self.video_panel = tk.Label(self.root, borderwidth=2, relief="solid")
        self.video_panel.place(x=50, y=50, width=640, height=480)
        
        # Skeleton display
        self.skeleton_panel = tk.Label(self.root, borderwidth=2, relief="solid", bg='white')
        self.skeleton_panel.place(x=750, y=50, width=400, height=400)
        
        # Prediction display
        self.prediction_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.prediction_frame.place(x=750, y=470, width=400, height=100)
        
        tk.Label(self.prediction_frame, text="Predicted:", 
                font=("Helvetica", 16), bg='#f0f0f0').pack(side=tk.LEFT)
        
        self.char_display = tk.Label(self.prediction_frame, text=" ", 
                                   font=("Helvetica", 24, "bold"), 
                                   bg='#f0f0f0', fg='blue', width=3)
        self.char_display.pack(side=tk.LEFT, padx=10)
        
        tk.Label(self.prediction_frame, text="Confidence:", 
                font=("Helvetica", 16), bg='#f0f0f0').pack(side=tk.LEFT)
        
        self.conf_display = tk.Label(self.prediction_frame, text="0.00", 
                                   font=("Helvetica", 16), 
                                   bg='#f0f0f0', fg='green')
        self.conf_display.pack(side=tk.LEFT, padx=10)
        
        # Sentence display
        self.sentence_frame = tk.Frame(self.root, bg='white', borderwidth=2, relief="solid")
        self.sentence_frame.place(x=50, y=550, width=640, height=80)
        
        self.sentence_display = tk.Label(self.sentence_frame, text=" ", 
                                       font=("Helvetica", 24), 
                                       bg='white', wraplength=600)
        self.sentence_display.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        self.button_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.button_frame.place(x=750, y=600, width=400, height=100)
        
        self.add_btn = tk.Button(self.button_frame, text="Add Letter", 
                               font=("Helvetica", 14), 
                               command=self.add_letter)
        self.add_btn.pack(side=tk.LEFT, padx=5)
        
        self.space_btn = tk.Button(self.button_frame, text="Space", 
                                 font=("Helvetica", 14), 
                                 command=self.add_space)
        self.space_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(self.button_frame, text="Clear", 
                                 font=("Helvetica", 14), 
                                 command=self.clear_sentence)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.speak_btn = tk.Button(self.button_frame, text="Speak", 
                                 font=("Helvetica", 14), 
                                 command=self.speak_sentence)
        self.speak_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_label = tk.Label(self.root, text="Waiting for hand...", 
                                   font=("Helvetica", 12), fg='gray')
        self.status_label.place(x=50, y=640)
        
        # Initialize text-to-speech
        try:
            self.speak_engine = pyttsx3.init()
            self.speak_engine.setProperty("rate", 150)
            voices = self.speak_engine.getProperty("voices")
            self.speak_engine.setProperty("voice", voices[0].id)
            self.debug("Text-to-speech initialized")
        except Exception as e:
            self.debug(f"ERROR initializing TTS: {str(e)}")
        
        self.current_symbol = " "
        self.sentence = " "
        self.confidence = 0.0
        self.hand_detected = False
        self.debug("UI setup complete")

    def update_frame(self):
        """Process each video frame"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.debug("Failed to read frame from camera")
                return
                
            frame = cv2.flip(frame, 1)
            hands = self.hd.findHands(frame, draw=False, flipType=True)
            
            # Reset white background
            self.white_bg = np.ones((400, 400, 3), np.uint8) * 255
            self.hand_detected = False
            
            if hands:
                hand = hands[0]
                if 'bbox' in hand:
                    x, y, w, h = hand['bbox']
                    self.debug(f"Hand detected at ({x},{y}) size {w}x{h}")
                    
                    # Ensure we don't go outside the frame
                    y1 = max(0, y-self.offset)
                    y2 = min(frame.shape[0], y+h+self.offset)
                    x1 = max(0, x-self.offset)
                    x2 = min(frame.shape[1], x+w+self.offset)
                    
                    image = frame[y1:y2, x1:x2]
                    
                    if image.size > 0:
                        handz = self.hd2.findHands(image, draw=False, flipType=True)
                        
                        if handz:
                            hand = handz[0]
                            if 'lmList' in hand:
                                pts = hand['lmList']
                                self.debug(f"Found {len(pts)} landmarks")
                                self.draw_skeleton(pts, w, h)
                                self.process_prediction()
                                self.hand_detected = True
            else:
                self.debug("No hands detected")
            
            # Update status
            status_text = "Hand detected" if self.hand_detected else "Waiting for hand..."
            status_color = "green" if self.hand_detected else "gray"
            self.status_label.config(text=status_text, fg=status_color)
            
            # Display video feed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk
            self.video_panel.config(image=imgtk)
            
            # Display skeleton
            skeleton_img = Image.fromarray(self.white_bg)
            skeleton_tk = ImageTk.PhotoImage(image=skeleton_img)
            self.skeleton_panel.imgtk = skeleton_tk
            self.skeleton_panel.config(image=skeleton_tk)
            
        except Exception as e:
            self.debug(f"ERROR in frame processing: {str(e)}")
            print(traceback.format_exc())
        finally:
            self.root.after(30, self.update_frame)  # Reduced frame rate for stability

    def draw_skeleton(self, pts, w, h):
        """Draw hand skeleton on white background"""
        try:
            os = ((w- 400) // 2) - 15
            os1 = ((h-400) // 2) - 15
            
            # Draw connections between landmarks
            connections = [
                (0, 1, 2, 3, 4),       # Thumb
                (0, 5, 6, 7, 8),        # Index
                (0, 9, 10, 11, 12),     # Middle
                (0, 13, 14, 15, 16),    # Ring
                (0, 17, 18, 19, 20),    # Pinky
                (5, 9, 13, 17)          # Palm
            ]
            
            for connection in connections:
                for i in range(len(connection)-1):
                    cv2.line(self.white_bg, 
                            (pts[connection[i]][0]+os, pts[connection[i]][1]+os1),
                            (pts[connection[i+1]][0]+os, pts[connection[i+1]][1]+os1),
                            (0, 255, 0), 3)
            
            # Draw landmarks
            for i in range(21):
                cv2.circle(self.white_bg, 
                          (pts[i][0]+os, pts[i][1]+os1), 
                          4, (0, 0, 255), -1)
            
            self.debug("Skeleton drawn successfully")
        except Exception as e:
            self.debug(f"ERROR drawing skeleton: {str(e)}")

    def process_prediction(self):
        """Process the hand skeleton for prediction"""
        try:
            # Preprocess image
            gray = cv2.cvtColor(self.white_bg, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (55, 55), interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0
            input_img = np.expand_dims(normalized, axis=(0, -1))
            
            # Make prediction
            predictions = self.model.predict(input_img)
            predicted_class = np.argmax(predictions[0])
            self.current_symbol = self.class_names[predicted_class]
            self.confidence = np.max(predictions[0])
            
            # Update displays
            self.char_display.config(text=self.current_symbol)
            self.conf_display.config(text=f"{self.confidence:.2f}",
                                   fg='green' if self.confidence > 0.7 else 
                                      'orange' if self.confidence > 0.4 else 
                                      'red')
            
            self.debug(f"Predicted: {self.current_symbol} (Confidence: {self.confidence:.2f})")
            
        except Exception as e:
            self.debug(f"ERROR in prediction: {str(e)}")
            self.current_symbol = " "
            self.confidence = 0.0

    def add_letter(self):
        """Add current letter to sentence"""
        if self.current_symbol and self.hand_detected:
            self.sentence += self.current_symbol
            self.sentence_display.config(text=self.sentence)
            self.debug(f"Added '{self.current_symbol}' to sentence")

    def add_space(self):
        """Add space to sentence"""
        self.sentence += " "
        self.sentence_display.config(text=self.sentence)
        self.debug("Added space to sentence")

    def clear_sentence(self):
        """Clear the sentence"""
        self.sentence = " "
        self.sentence_display.config(text=self.sentence)
        self.debug("Cleared sentence")

    def speak_sentence(self):
        """Convert sentence to speech"""
        if len(self.sentence.strip()) > 0:
            self.speak_engine.say(self.sentence)
            self.speak_engine.runAndWait()
            self.debug(f"Spoken sentence: {self.sentence}")

    def on_closing(self):
        """Clean up on window close"""
        self.debug("Closing application...")
        self.cap.release()
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
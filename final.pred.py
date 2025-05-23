'''
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from collections import defaultdict
import pyttsx3  # Text-to-speech library
import time

class ASLRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Recognition System")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # Load model and initialize variables
        self.model = load_model('best_model.h5')
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # Cooldown parameters
        self.letter_cooldown = 1.5  # seconds between letters
        self.last_letter_time = 0
        self.cooldown_active = False
        
        # Gesture recognition parameters
        self.space_gesture_threshold = 0.8  # Minimum confidence for space gesture
        self.backspace_gesture_threshold = 0.8  # Minimum confidence for backspace gesture
        self.gesture_hold_frames = 15  # Number of frames to hold gesture before action
        self.space_gesture_counter = 0
        self.backspace_gesture_counter = 0
        
        # Word dictionary for suggestions
        self.word_dictionary = {
            'A': ['Apple', 'Animal', 'Amazing', 'Airplane', 'Adventure'],
            'B': ['Banana', 'Beautiful', 'Butterfly', 'Basketball', 'Brother'],
            'C': ['Cat', 'Computer', 'Chocolate', 'Camera', 'Candle'],
            'D': ['Dog', 'Dolphin', 'Diamond', 'Dinosaur', 'Dragon'],
            'E': ['Elephant', 'Excellent', 'Envelope', 'Emerald', 'Energy'],
            'F': ['Flower', 'Football', 'Friendship', 'Fireplace', 'Fantastic'],
            'G': ['Giraffe', 'Garden', 'Goldfish', 'Galaxy', 'Guitar'],
            'H': ['House', 'Happiness', 'Helicopter', 'Hamburger', 'Holiday'],
            'I': ['Icecream', 'Island', 'Internet', 'Important', 'Incredible'],
            'J': ['Jellyfish', 'Jungle', 'Journey', 'Jacket', 'Jigsaw'],
            'K': ['Kangaroo', 'Kitchen', 'Keyboard', 'Kitten', 'Knowledge'],
            'L': ['Lion', 'Lollipop', 'Lighthouse', 'Library', 'Lemonade'],
            'M': ['Monkey', 'Mountain', 'Mystery', 'Magazine', 'Moonlight'],
            'N': ['Notebook', 'Nighttime', 'Necklace', 'Noodles', 'Natural'],
            'O': ['Orange', 'Octopus', 'Ocean', 'Orchestra', 'Outstanding'],
            'P': ['Pineapple', 'Penguin', 'Puzzle', 'Paradise', 'Pancakes'],
            'Q': ['Queen', 'Quilt', 'Question', 'Quiet', 'Quick'],
            'R': ['Rainbow', 'Rocket', 'River', 'Restaurant', 'Rabbit'],
            'S': ['Sunshine', 'Strawberry', 'Snowman', 'Surprise', 'Soccer'],
            'T': ['Tiger', 'Telescope', 'Treasure', 'Tornado', 'Tomorrow'],
            'U': ['Umbrella', 'Universe', 'Unicorn', 'Unique', 'Upstairs'],
            'V': ['Violin', 'Vacation', 'Vegetable', 'Village', 'Volcano'],
            'W': ['Watermelon', 'Window', 'Wonderful', 'Whale', 'Wizard'],
            'X': ['Xylophone', 'Xray', 'Xmas', 'Xenon', 'Xerox'],
            'Y': ['Yacht', 'Yellow', 'Yogurt', 'Yesterday', 'Yummy'],
            'Z': ['Zebra', 'Zoo', 'Zipper', 'Zombie', 'Zodiac']
        }
        
        # Video capture setup
        self.capture = cv2.VideoCapture(0)
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.offset = 15
        
        # Sentence formation
        self.sentence = ""
        self.last_letter = None
        self.letter_count = 0
        self.threshold = 0.80
        self.current_suggestions = []
        
        # Create GUI elements
        self.setup_gui()
        
        # Start video processing thread
        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()
        
        # White background for skeleton (3-channel color image)
        self.white = np.ones((400, 400, 3), np.uint8) * 255
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def setup_gui(self):
        # Main frames
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.skeleton_frame = ttk.LabelFrame(self.root, text="Hand Skeleton")
        self.skeleton_frame.grid(row=0, column=1, padx=10, pady=10)
        
        self.info_frame = ttk.LabelFrame(self.root, text="Recognition Info")
        self.info_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Video labels
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        self.skeleton_label = ttk.Label(self.skeleton_frame)
        self.skeleton_label.pack()
        
        # Info labels
        self.prediction_label = ttk.Label(self.info_frame, text="Predicted Letter: ", font=('Helvetica', 14))
        self.prediction_label.pack(anchor='w', padx=10, pady=5)
        
        self.confidence_label = ttk.Label(self.info_frame, text="Confidence: ", font=('Helvetica', 14))
        self.confidence_label.pack(anchor='w', padx=10, pady=5)
        
        # Gesture status label
        self.gesture_label = ttk.Label(self.info_frame, text="Gesture: ", font=('Helvetica', 14))
        self.gesture_label.pack(anchor='w', padx=10, pady=5)
        
        # Word suggestions frame
        self.suggestions_frame = ttk.LabelFrame(self.info_frame, text="Word Suggestions")
        self.suggestions_frame.pack(anchor='w', padx=10, pady=5, fill='x')
        
        self.suggestion_buttons = []
        for i in range(5):
            btn = ttk.Button(self.suggestions_frame, text="", 
                            command=lambda i=i: self.add_suggestion_to_sentence(i))
            btn.pack(side='left', padx=5, pady=5)
            self.suggestion_buttons.append(btn)
        
        # Sentence display
        self.sentence_frame = ttk.Frame(self.info_frame)
        self.sentence_frame.pack(fill='x', padx=10, pady=10)
        
        self.sentence_label = ttk.Label(self.sentence_frame, text="Sentence: ", font=('Helvetica', 16))
        self.sentence_label.pack(side='left', anchor='w')
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.info_frame)
        self.control_frame.pack(side='right', padx=10, pady=10)
        
        # Text-to-speech button
        self.speak_button = ttk.Button(self.control_frame, text="Speak", command=self.speak_sentence)
        self.speak_button.pack(side='left', padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(self.control_frame, text="Clear", command=self.clear_sentence)
        self.clear_button.pack(side='left', padx=5)
        
        # Space button
        self.space_button = ttk.Button(self.control_frame, text="Space", command=self.add_space)
        self.space_button.pack(side='left', padx=5)
        
        # Backspace button
        self.backspace_button = ttk.Button(self.control_frame, text="Backspace", command=self.backspace)
        self.backspace_button.pack(side='left', padx=5)
        
        # Threshold control
        self.threshold_frame = ttk.Frame(self.info_frame)
        self.threshold_frame.pack(anchor='w', padx=10, pady=5)
        
        ttk.Label(self.threshold_frame, text="Confidence Threshold:").pack(side='left')
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0.5, to=1.0, value=0.8, 
                                        command=self.update_threshold)
        self.threshold_slider.pack(side='left', padx=5)
        self.threshold_value = ttk.Label(self.threshold_frame, text="0.80")
        self.threshold_value.pack(side='left')
    
    def update_threshold(self, value):
        self.threshold = float(value)
        self.threshold_value.config(text=f"{self.threshold:.2f}")
    
    def clear_sentence(self):
        self.sentence = ""
        self.sentence_label.config(text="Sentence: ")
        self.last_letter = None
        self.letter_count = 0
        self.engine.say("Sentence cleared")
        self.engine.runAndWait()
    
    def add_space(self):
        self.sentence += " "
        self.update_sentence_display()
        self.engine.say("Space added")
        self.engine.runAndWait()
    
    def backspace(self):
        if len(self.sentence) > 0:
            self.sentence = self.sentence[:-1]
            self.update_sentence_display()
            self.engine.say("Backspace")
            self.engine.runAndWait()
    
    def speak_sentence(self):
        if self.sentence.strip():
            self.engine.say(self.sentence)
            self.engine.runAndWait()
        else:
            self.engine.say("No sentence to speak")
            self.engine.runAndWait()
    
    def update_sentence_display(self):
        self.sentence_label.config(text=f"Sentence: {self.sentence}")
    
    def add_suggestion_to_sentence(self, index):
        if index < len(self.current_suggestions):
            self.sentence += self.current_suggestions[index] + " "
            self.update_sentence_display()
            self.last_letter = None
            self.letter_count = 0
            self.engine.say(f"Added {self.current_suggestions[index]}")
            self.engine.runAndWait()
    
    def update_suggestions(self, letter):
        if letter in self.word_dictionary:
            self.current_suggestions = self.word_dictionary[letter]
            for i in range(5):
                if i < len(self.current_suggestions):
                    self.suggestion_buttons[i].config(text=self.current_suggestions[i])
                else:
                    self.suggestion_buttons[i].config(text="")
        else:
            self.current_suggestions = []
            for btn in self.suggestion_buttons:
                btn.config(text="")
    
    def detect_gestures(self, hand):
        """Detect special gestures for space and backspace"""
        fingers = self.hd2.fingersUp(hand)
        
        # Space gesture: All fingers extended (open hand)
        if sum(fingers) == 5:  # All fingers up
            self.space_gesture_counter += 1
            self.backspace_gesture_counter = 0
            if self.space_gesture_counter >= self.gesture_hold_frames:
                self.space_gesture_counter = 0
                return "space"
        # New backspace gesture: Only thumb extended (thumb out)
        elif fingers == [1, 0, 0, 0, 0]:  # Only thumb is up
            self.backspace_gesture_counter += 1
            self.space_gesture_counter = 0
            if self.backspace_gesture_counter >= self.gesture_hold_frames:
                self.backspace_gesture_counter = 0
                return "backspace"
        else:
            self.space_gesture_counter = 0
            self.backspace_gesture_counter = 0
            
        return None
    
    def process_video(self):
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                hands, _ = self.hd.findHands(frame, draw=False, flipType=True)
                white = self.white.copy()
                predicted_letter = None
                confidence = 0
                current_gesture = None
                
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    image = np.array(frame[y - self.offset:y + h + self.offset, 
                                      x - self.offset:x + w + self.offset])
                    
                    # Make sure the cropped image is valid
                    if image.size == 0:
                        continue
                        
                    handz, imz = self.hd2.findHands(image, draw=True, flipType=True)
                    
                    if handz:
                        hand = handz[0]
                        pts = hand['lmList']
                        os = ((400 - w) // 2) - 15
                        os1 = ((400 - h) // 2) - 15
                        
                        # Check for special gestures first
                        gesture = self.detect_gestures(hand)
                        
                        if gesture == "space":
                            self.add_space()
                            current_gesture = "SPACE GESTURE (OPEN HAND)"
                        elif gesture == "backspace":
                            self.backspace()
                            current_gesture = "BACKSPACE GESTURE (THUMB OUT)"
                        else:
                            # Draw skeleton lines
                            connections = [
                                (0, 1, 2, 3, 4),       # Thumb
                                (0, 5, 6, 7, 8),       # Index
                                (0, 9, 10, 11, 12),    # Middle
                                (0, 13, 14, 15, 16),   # Ring
                                (0, 17, 18, 19, 20),   # Pinky
                                (5, 9, 13, 17)         # Palm connections
                            ]
                            
                            for connection in connections:
                                for i in range(len(connection)-1):
                                    cv2.line(white, 
                                            (pts[connection[i]][0]+os, pts[connection[i]][1]+os1),
                                            (pts[connection[i+1]][0]+os, pts[connection[i+1]][1]+os1),
                                            (0, 255, 0), 3)
                            
                            # Draw landmarks
                            for i in range(21):
                                cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0, 0, 255), 1)
                            
                            # Convert to grayscale if needed (but ensure it's a 3-channel image first)
                            if len(white.shape) == 2:
                                white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
                            
                            # Preprocess for model prediction
                            gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (55, 55), interpolation=cv2.INTER_AREA)
                            normalized = resized / 255.0
                            input_img = np.expand_dims(normalized, axis=(0, -1))
                            
                            predictions = self.model.predict(input_img)
                            predicted_class = np.argmax(predictions[0])
                            predicted_letter = self.class_names[predicted_class]
                            confidence = np.max(predictions[0])
                            
                            # Update word suggestions
                            self.update_suggestions(predicted_letter)
                            
                            # Add to sentence if confidence is high enough
                            if confidence >= self.threshold:
                                if predicted_letter != self.last_letter:
                                    self.last_letter = predicted_letter
                                    self.letter_count = 1
                                else:
                                    self.letter_count += 1
                                    
                                # Add to sentence after holding the letter for a few frames
                                if self.letter_count == 10:  # Adjust this value for sensitivity
                                    self.sentence += predicted_letter
                                    self.update_sentence_display()
                                    self.engine.say(predicted_letter)
                                    self.engine.runAndWait()
                                    self.last_letter = None
                                    self.letter_count = 0
                else:
                    # Clear suggestions when no hand is detected
                    self.update_suggestions(None)
                    self.space_gesture_counter = 0
                    self.backspace_gesture_counter = 0
                
                # Update GUI
                self.update_frames(frame, white, predicted_letter, confidence, current_gesture)
                
            except Exception as e:
                print("Error:", traceback.format_exc())
                continue
    
    def update_frames(self, frame, skeleton, letter=None, confidence=0, gesture=None):
        # Convert frames to PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame)
        frame_img = ImageTk.PhotoImage(image=frame_img)
        
        # Ensure skeleton is 3-channel for display
        if len(skeleton.shape) == 2:
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        else:
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
            
        skeleton_img = Image.fromarray(skeleton)
        skeleton_img = ImageTk.PhotoImage(image=skeleton_img)
        
        # Update labels
        self.video_label.config(image=frame_img)
        self.video_label.image = frame_img
        
        self.skeleton_label.config(image=skeleton_img)
        self.skeleton_label.image = skeleton_img
        
        # Update prediction info
        if letter:
            self.prediction_label.config(text=f"Predicted Letter: {letter}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
        else:
            self.prediction_label.config(text="Predicted Letter: ")
            self.confidence_label.config(text="Confidence: ")
        
        # Update gesture info
        if gesture:
            self.gesture_label.config(text=f"Gesture: {gesture}")
        else:
            self.gesture_label.config(text="Gesture: ")
    
    def on_close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLRecognitionApp(root)
    root.mainloop()

'''
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from collections import defaultdict
import pyttsx3  # Text-to-speech library
import nltk
from nltk.corpus import words
from difflib import get_close_matches
import time

class ASLRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Recognition System")
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        
        # Initialize NLTK
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words', quiet=True)
        self.english_words = set(words.words())
        
        # Load model and initialize variables
        self.model = load_model('best_model.h5')
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        # Cooldown parameters
        self.letter_cooldown = 1.5  # seconds between letters
        self.last_letter_time = 0
        self.cooldown_active = False
        
        # Gesture recognition parameters
        self.space_gesture_threshold = 0.8  # Minimum confidence for space gesture
        self.backspace_gesture_threshold = 0.8  # Minimum confidence for backspace gesture
        self.gesture_hold_frames = 15  # Number of frames to hold gesture before action
        self.space_gesture_counter = 0
        self.backspace_gesture_counter = 0
        
        # Initialize word suggestions cache
        self.suggestions_cache = {}
        
        # Video capture setup
        self.capture = cv2.VideoCapture(0)
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.offset = 15
        
        # Sentence formation
        self.sentence = ""
        self.last_letter = None
        self.letter_count = 0
        self.threshold = 0.80
        self.current_suggestions = []
        
        # Create GUI elements
        self.setup_gui()
        
        # Start video processing thread
        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()
        
        # White background for skeleton (3-channel color image)
        self.white = np.ones((400, 400, 3), np.uint8) * 255
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def setup_gui(self):
        # Main frames
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        
        self.skeleton_frame = ttk.LabelFrame(self.root, text="Hand Skeleton")
        self.skeleton_frame.grid(row=0, column=1, padx=10, pady=10)
        
        self.info_frame = ttk.LabelFrame(self.root, text="Recognition Info")
        self.info_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Video labels
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        self.skeleton_label = ttk.Label(self.skeleton_frame)
        self.skeleton_label.pack()
        
        # Info labels
        self.prediction_label = ttk.Label(self.info_frame, text="Predicted Letter: ", font=('Helvetica', 14))
        self.prediction_label.pack(anchor='w', padx=10, pady=5)
        
        self.confidence_label = ttk.Label(self.info_frame, text="Confidence: ", font=('Helvetica', 14))
        self.confidence_label.pack(anchor='w', padx=10, pady=5)
        
        # Gesture status label
        self.gesture_label = ttk.Label(self.info_frame, text="Gesture: ", font=('Helvetica', 14))
        self.gesture_label.pack(anchor='w', padx=10, pady=5)
        
        # Word suggestions frame
        self.suggestions_frame = ttk.LabelFrame(self.info_frame, text="Word Suggestions")
        self.suggestions_frame.pack(anchor='w', padx=10, pady=5, fill='x')
        
        self.suggestion_buttons = []
        for i in range(5):
            btn = ttk.Button(self.suggestions_frame, text="", 
                            command=lambda i=i: self.add_suggestion_to_sentence(i))
            btn.pack(side='left', padx=5, pady=5)
            self.suggestion_buttons.append(btn)
        
        # Sentence display
        self.sentence_frame = ttk.Frame(self.info_frame)
        self.sentence_frame.pack(fill='x', padx=10, pady=10)
        
        self.sentence_label = ttk.Label(self.sentence_frame, text="Sentence: ", font=('Helvetica', 16))
        self.sentence_label.pack(side='left', anchor='w')
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.info_frame)
        self.control_frame.pack(side='right', padx=10, pady=10)
        
        # Text-to-speech button
        self.speak_button = ttk.Button(self.control_frame, text="Speak", command=self.speak_sentence)
        self.speak_button.pack(side='left', padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(self.control_frame, text="Clear", command=self.clear_sentence)
        self.clear_button.pack(side='left', padx=5)
        
        # Space button
        self.space_button = ttk.Button(self.control_frame, text="Space", command=self.add_space)
        self.space_button.pack(side='left', padx=5)
        
        # Backspace button
        self.backspace_button = ttk.Button(self.control_frame, text="Backspace", command=self.backspace)
        self.backspace_button.pack(side='left', padx=5)
        
        # Threshold control
        self.threshold_frame = ttk.Frame(self.info_frame)
        self.threshold_frame.pack(anchor='w', padx=10, pady=5)
        
        ttk.Label(self.threshold_frame, text="Confidence Threshold:").pack(side='left')
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0.5, to=1.0, value=0.8, 
                                        command=self.update_threshold)
        self.threshold_slider.pack(side='left', padx=5)
        self.threshold_value = ttk.Label(self.threshold_frame, text="0.80")
        self.threshold_value.pack(side='left')
    
    def update_threshold(self, value):
        self.threshold = float(value)
        self.threshold_value.config(text=f"{self.threshold:.2f}")
    
    def clear_sentence(self):
        self.sentence = ""
        self.sentence_label.config(text="Sentence: ")
        self.last_letter = None
        self.letter_count = 0
        self.engine.say("Sentence cleared")
        self.engine.runAndWait()
    
    def add_space(self):
        self.sentence += " "
        self.update_sentence_display()
        self.engine.say("Space added")
        self.engine.runAndWait()
    
    def backspace(self):
        if len(self.sentence) > 0:
            self.sentence = self.sentence[:-1]
            self.update_sentence_display()
            self.engine.say("Backspace")
            self.engine.runAndWait()
    
    def speak_sentence(self):
        if self.sentence.strip():
            self.engine.say(self.sentence)
            self.engine.runAndWait()
        else:
            self.engine.say("No sentence to speak")
            self.engine.runAndWait()
    
    def update_sentence_display(self):
        self.sentence_label.config(text=f"Sentence: {self.sentence}")
    
    def add_suggestion_to_sentence(self, index):
        if index < len(self.current_suggestions):
            self.sentence += self.current_suggestions[index] + " "
            self.update_sentence_display()
            self.last_letter = None
            self.letter_count = 0
            self.engine.say(f"Added {self.current_suggestions[index]}")
            self.engine.runAndWait()
    
    def update_suggestions(self, letter):
        if not letter:
            self.current_suggestions = []
            for btn in self.suggestion_buttons:
                btn.config(text="")
            return

        # Check cache first
        if letter in self.suggestions_cache:
            self.current_suggestions = self.suggestions_cache[letter]
        else:
            # Find words that start with the letter
            prefix_matches = [word.capitalize() for word in self.english_words 
                            if word.lower().startswith(letter.lower())]
            
            # Sort by length and get most common/shorter words first
            prefix_matches = sorted(prefix_matches, key=len)
            
            # If we don't have enough prefix matches, get close matches
            if len(prefix_matches) < 5:
                close_matches = [match.capitalize() for match in 
                               get_close_matches(letter.lower(), 
                                                [word.lower() for word in self.english_words],
                                                n=5, cutoff=0.6)]
                # Combine and remove duplicates
                all_matches = list(dict.fromkeys(prefix_matches + close_matches))
                self.current_suggestions = all_matches[:5]
            else:
                self.current_suggestions = prefix_matches[:5]
            
            # Cache the results
            self.suggestions_cache[letter] = self.current_suggestions
        
        # Update buttons
        for i in range(5):
            if i < len(self.current_suggestions):
                self.suggestion_buttons[i].config(text=self.current_suggestions[i])
            else:
                self.suggestion_buttons[i].config(text="")
    
    def detect_gestures(self, hand):
        """Detect special gestures for space and backspace"""
        fingers = self.hd2.fingersUp(hand)
        
        # Space gesture: All fingers extended (open hand)
        if sum(fingers) == 5:  # All fingers up
            self.space_gesture_counter += 1
            self.backspace_gesture_counter = 0
            if self.space_gesture_counter >= self.gesture_hold_frames:
                self.space_gesture_counter = 0
                return "space"
        # New backspace gesture: Only thumb extended (thumb out)
        elif fingers == [1, 0, 0, 0, 0]:  # Only thumb is up
            self.backspace_gesture_counter += 1
            self.space_gesture_counter = 0
            if self.backspace_gesture_counter >= self.gesture_hold_frames:
                self.backspace_gesture_counter = 0
                return "backspace"
        else:
            self.space_gesture_counter = 0
            self.backspace_gesture_counter = 0
            
        return None
    
    def process_video(self):
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                hands, _ = self.hd.findHands(frame, draw=False, flipType=True)
                white = self.white.copy()
                predicted_letter = None
                confidence = 0
                current_gesture = None
                
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    image = np.array(frame[y - self.offset:y + h + self.offset, 
                                      x - self.offset:x + w + self.offset])
                    
                    # Make sure the cropped image is valid
                    if image.size == 0:
                        continue
                        
                    handz, imz = self.hd2.findHands(image, draw=True, flipType=True)
                    
                    if handz:
                        hand = handz[0]
                        pts = hand['lmList']
                        os = ((400 - w) // 2) - 15
                        os1 = ((400 - h) // 2) - 15
                        
                        # Check for special gestures first
                        gesture = self.detect_gestures(hand)
                        
                        if gesture == "space":
                            self.add_space()
                            current_gesture = "SPACE GESTURE (OPEN HAND)"
                        elif gesture == "backspace":
                            self.backspace()
                            current_gesture = "BACKSPACE GESTURE (THUMB OUT)"
                        else:
                            # Draw skeleton lines
                            connections = [
                                (0, 1, 2, 3, 4),       # Thumb
                                (0, 5, 6, 7, 8),       # Index
                                (0, 9, 10, 11, 12),    # Middle
                                (0, 13, 14, 15, 16),   # Ring
                                (0, 17, 18, 19, 20),   # Pinky
                                (5, 9, 13, 17)         # Palm connections
                            ]
                            
                            for connection in connections:
                                for i in range(len(connection)-1):
                                    cv2.line(white, 
                                            (pts[connection[i]][0]+os, pts[connection[i]][1]+os1),
                                            (pts[connection[i+1]][0]+os, pts[connection[i+1]][1]+os1),
                                            (0, 255, 0), 3)
                            
                            # Draw landmarks
                            for i in range(21):
                                cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0, 0, 255), 1)
                            
                            # Convert to grayscale if needed (but ensure it's a 3-channel image first)
                            if len(white.shape) == 2:
                                white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
                            
                            # Preprocess for model prediction
                            gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (55, 55), interpolation=cv2.INTER_AREA)
                            normalized = resized / 255.0
                            input_img = np.expand_dims(normalized, axis=(0, -1))
                            
                            predictions = self.model.predict(input_img)
                            predicted_class = np.argmax(predictions[0])
                            predicted_letter = self.class_names[predicted_class]
                            confidence = np.max(predictions[0])
                            
                            # Update word suggestions
                            self.update_suggestions(predicted_letter)
                            
                            # Add to sentence if confidence is high enough
                            if confidence >= self.threshold:
                                if predicted_letter != self.last_letter:
                                    self.last_letter = predicted_letter
                                    self.letter_count = 1
                                else:
                                    self.letter_count += 1
                                    
                                # Add to sentence after holding the letter for a few frames
                                if self.letter_count == 10:  # Adjust this value for sensitivity
                                    self.sentence += predicted_letter
                                    self.update_sentence_display()
                                    self.engine.say(predicted_letter)
                                    self.engine.runAndWait()
                                    self.last_letter = None
                                    self.letter_count = 0
                else:
                    # Clear suggestions when no hand is detected
                    self.update_suggestions(None)
                    self.space_gesture_counter = 0
                    self.backspace_gesture_counter = 0
                
                # Update GUI
                self.update_frames(frame, white, predicted_letter, confidence, current_gesture)
                
            except Exception as e:
                print("Error:", traceback.format_exc())
                continue
    
    def update_frames(self, frame, skeleton, letter=None, confidence=0, gesture=None):
        # Convert frames to PhotoImage
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame)
        frame_img = ImageTk.PhotoImage(image=frame_img)
        
        # Ensure skeleton is 3-channel for display
        if len(skeleton.shape) == 2:
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        else:
            skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
            
        skeleton_img = Image.fromarray(skeleton)
        skeleton_img = ImageTk.PhotoImage(image=skeleton_img)
        
        # Update labels
        self.video_label.config(image=frame_img)
        self.video_label.image = frame_img
        
        self.skeleton_label.config(image=skeleton_img)
        self.skeleton_label.image = skeleton_img
        
        # Update prediction info
        if letter:
            self.prediction_label.config(text=f"Predicted Letter: {letter}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
        else:
            self.prediction_label.config(text="Predicted Letter: ")
            self.confidence_label.config(text="Confidence: ")
        
        # Update gesture info
        if gesture:
            self.gesture_label.config(text=f"Gesture: {gesture}")
        else:
            self.gesture_label.config(text="Gesture: ")
    
    def on_close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLRecognitionApp(root)
    root.mainloop()
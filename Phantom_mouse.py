import cv2 #for image and video processing.
import mediapipe as mp # real-time hand tracking.
import pyautogui #mouse control
import random #Generates random numbers (used for naming screenshots)
import numpy as np #Handles mathematical operations like angle calculations
from pynput.mouse import Button, Controller #mouse click action
import threading #for running video processing in a separate thread.
#Runs video processing in a separate thread (so it doesn’t block the UI)
import tkinter as tk
from tkinter import messagebox

mouse = Controller() #Initializes mouse for controlling mouse clicks
screen_width, screen_height = pyautogui.size() #Gets the screen width and height for mapping hand gestures

mpHands = mp.solutions.hands #Stores the hand tracking model
hands = mpHands.Hands( #Initializes hand tracking
    static_image_mode=False, #Uses video instead of still images
    model_complexity=1, #Sets model accuracy (higher = slower but better tracking)
    min_detection_confidence=0.7, #The minimum confidence level for detecting a hand
    min_tracking_confidence=0.7, #Confidence for tracking movements
    max_num_hands=1 #Tracks only one hand
)

running = False #Boolean flag to start/stop the video stream
cap = None #Stores the video capture object

def get_angle(a, b, c): #Used to detect finger bending
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle

def get_distance(landmark_ist):
    if len(landmark_ist) < 2:
        return
    (x1, y1), (x2, y2) = landmark_ist[0], landmark_ist[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None

def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )
#Checks if index finger bends (< 50 degrees).

#Checks if middle finger is straight (> 90 degrees).

#Ensures thumb is far away (> 50 distance)
def is_right_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 50
    )
#Checks middle finger bends (< 50 degrees).

#Checks index finger is straight (> 90 degrees).

#Ensures thumb is far away
def is_double_click(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50
    )
#Checks both index and middle fingers are bent
def is_screenshot(landmark_list, thumb_index_dist):
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )
#Takes a screenshot if fingers are bent and thumb is close
def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        if thumb_index_dist < 50 and get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def start_video():
    global running, cap
    if running:
        return
    running = True
    cap = cv2.VideoCapture(0)

    new_width = 1000
    new_height = 700
    draw = mp.solutions.drawing_utils

    def video_loop():
        global running
        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)


            frame = cv2.resize(frame, (new_width, new_height))

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=video_loop).start()

def stop_video():
    global running
    running = False
    messagebox.showinfo("Info", "Video Stopped")

root = tk.Tk()
root.title("Hand Gesture Control")
root.geometry("400x300")

start_button = tk.Button(root, text="Start", command=start_video, width=15, height=2)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop", command=stop_video, width=15, height=2)
stop_button.pack(pady=10)

root.mainloop()

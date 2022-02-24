import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# Window size
frame_size_x = 1080
frame_size_y = 720

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(
        f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Digital Art')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()


def mark_landmark(landmark_data, size=10):
    landmark_x = frame_size_x - int(landmark_data.x * frame_size_x)
    landmark_y = int(landmark_data.y * frame_size_y)
    landmark_z = 1  # landmark_data.z
    pygame.draw.circle(game_window, white, [
                       landmark_x, landmark_y], size*landmark_z)


def line_between_landmarks(landmark_data_1, landmark_data_2, size=5):
    landmark_1_x = frame_size_x - int(landmark_data_1.x * frame_size_x)
    landmark_1_y = int(landmark_data_1.y * frame_size_y)
    landmark_2_x = frame_size_x - int(landmark_data_2.x * frame_size_x)
    landmark_2_y = int(landmark_data_2.y * frame_size_y)
    pygame.draw.line(game_window, white, [landmark_1_x, landmark_1_y], [
        landmark_2_x, landmark_2_y], size)


def mark_between_landmarks(landmark_data_1, landmark_data_2, size=10):
    landmark_1_x = frame_size_x - int(landmark_data_1.x * frame_size_x)
    landmark_1_y = int(landmark_data_1.y * frame_size_y)
    landmark_2_x = frame_size_x - int(landmark_data_2.x * frame_size_x)
    landmark_2_y = int(landmark_data_2.y * frame_size_y)

    pygame.draw.circle(game_window, white, [
                       (landmark_1_x + landmark_2_x)/2, (landmark_1_y + landmark_2_y)/2], size)


def draw_neck(nose_data, left_shoulder_data, right_shoulder_data):
    nose_x = frame_size_x - int(nose_data.x * frame_size_x)
    nose_y = int(nose_data.y * frame_size_y)
    left_shoulder_x = frame_size_x - int(left_shoulder_data.x * frame_size_x)
    left_shoulder_y = int(left_shoulder_data.y * frame_size_y)
    right_shoulder_x = frame_size_x - int(right_shoulder_data.x * frame_size_x)
    right_shoulder_y = int(right_shoulder_data.y * frame_size_y)

    pygame.draw.line(game_window, white, [nose_x, nose_y], [
        (left_shoulder_x + right_shoulder_x)/2, (left_shoulder_y + right_shoulder_y)/2], 5)


def draw_pp(left_hip_data, right_hip_data):
    left_hip_x = frame_size_x - int(left_hip_data.x * frame_size_x)
    left_hip_y = int(left_hip_data.y * frame_size_y)
    left_hip_visibility = left_hip_data.visibility
    right_hip_x = frame_size_x - int(right_hip_data.x * frame_size_x)
    right_hip_y = int(right_hip_data.y * frame_size_y)
    right_hip_visibility = right_hip_data.visibility

    pp_x = (left_hip_x + right_hip_x)/2
    pp_y = (left_hip_y + right_hip_y)/2

    pp_dir = -frame_size_x/5
    if (left_hip_visibility > right_hip_visibility):
        pp_dir = frame_size_x/5

    if (abs(left_hip_x - right_hip_x) < frame_size_x/20):
        pygame.draw.line(game_window, white, [pp_x, pp_y], [
            pp_x + pp_dir, pp_y], 15)


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Display the game
        game_window.fill(black)
        # Check for landmarks
        try:
            nose_data = results.pose_landmarks.landmark[0]
            left_hip_data = results.pose_landmarks.landmark[23]
            right_hip_data = results.pose_landmarks.landmark[24]
            left_shoulder_data = results.pose_landmarks.landmark[11]
            right_shoulder_data = results.pose_landmarks.landmark[12]
            left_elbow_data = results.pose_landmarks.landmark[13]
            right_elbow_data = results.pose_landmarks.landmark[14]
            left_wrist_data = results.pose_landmarks.landmark[15]
            right_wrist_data = results.pose_landmarks.landmark[16]
            left_knee_data = results.pose_landmarks.landmark[25]
            right_knee_data = results.pose_landmarks.landmark[26]
            left_ankle_data = results.pose_landmarks.landmark[27]
            right_ankle_data = results.pose_landmarks.landmark[28]

            # mark landmarks
            mark_landmark(nose_data, 30)
            mark_landmark(left_hip_data)
            mark_landmark(right_hip_data)
            mark_landmark(left_shoulder_data)
            mark_landmark(right_shoulder_data)
            mark_landmark(left_wrist_data)
            mark_landmark(right_wrist_data)
            mark_landmark(left_ankle_data)
            mark_landmark(right_ankle_data)
            mark_between_landmarks(left_hip_data, right_hip_data, 15)  # pp

            # mark edges
            line_between_landmarks(left_hip_data, right_hip_data)
            line_between_landmarks(left_shoulder_data, right_shoulder_data)
            line_between_landmarks(left_shoulder_data, left_elbow_data)
            line_between_landmarks(right_shoulder_data, right_elbow_data)
            line_between_landmarks(left_elbow_data, left_wrist_data)
            line_between_landmarks(right_elbow_data, right_wrist_data)
            line_between_landmarks(left_shoulder_data, left_hip_data)
            line_between_landmarks(right_shoulder_data, right_hip_data)
            line_between_landmarks(left_hip_data, left_knee_data)
            line_between_landmarks(right_hip_data, right_knee_data)
            line_between_landmarks(left_knee_data, left_ankle_data)
            line_between_landmarks(right_knee_data, right_ankle_data)

            draw_neck(nose_data, left_shoulder_data, right_shoulder_data)
            draw_pp(left_hip_data, right_hip_data)

        # If no landmarks are detected, set variables to default values
        except:
            a = 0

        pygame.display.update()

cap.release()

# x: 0.5849112272262573
# y: 0.5975667238235474
# z: -0.8629480600357056
# visibility: 0.9998940825462341

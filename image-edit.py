import os
import cv2
import dlib
import numpy as np

# Set the path to the folder containing the images
folder_path = "<path of dir containing pictures>"

# Create an instance of the dlib facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Iterate through the images in the folder
for file in os.listdir(folder_path):
    # Load the image
    img = cv2.imread(os.path.join(folder_path, file))

    # Detect faces in the image
    faces = detector(img)

    for face in faces:
        # Get facial landmarks for the face
        landmarks = predictor(img, face)

        # Get the coordinates for the left and right eyes
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y

        # Calculate the center point between the eyes
        eyes_center = (left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2

        # Calculate the angle between the eyes
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate the image to align the eyes
        img = imutils.rotate(img, angle)

        # Crop the image to remove the background
        img = img[0: img.shape[0], int(eyes_center[0] - img.shape[1] / 2):int(eyes_center[0] + img.shape[1] / 2)]

        # Apply a black background to the image
        img[:, :] = (0, 0, 0)

    # Save the aligned and background-covered image
    cv2.imwrite(os.path.join(folder_path, "aligned_" + file), img)

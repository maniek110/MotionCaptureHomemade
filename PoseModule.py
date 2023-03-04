import cv2
import sys
import time
import mediapipe as mp

class poseDetector():
    def __init__(self,
                 static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5
                 ):
        self.static_image_mode=static_image_mode
        self.upper_body_only=upper_body_only
        self.smooth_landmarks=smooth_landmarks
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
    def findPose(self,img,isDrawing=True):
        with self.mp_pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence) as pose:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img)

            # Draw the pose annotation on the image.
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            return img



def main():
    movie = 0
    if len(sys.argv) >= 2:
        movie = cv2.VideoCapture(sys.argv[1])
    else:
        movie = cv2.VideoCapture(0)
    detector=poseDetector()
    while movie.isOpened():
        success, image = movie.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        image=detector.findPose(image)
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ =="__main__":
    main()
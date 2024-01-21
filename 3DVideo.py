import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def main():
    cap = cv2.VideoCapture(0)  # You can change this to the video file path if you want to process a video.
    vertical_ref=205
    horizontal_ref=205 #change this wrt aruco
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and get the pose landmarks
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Extract landmark points
                head = (landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y)
                shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                waist = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                heel = (landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y)
                left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                left_wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
                right_wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)

                # Calculate distances
                head_to_heel = calculate_distance(head, heel)*vertical_ref
                shoulder_to_waist = calculate_distance(shoulder, waist)*vertical_ref
                waist_to_heel = calculate_distance(waist, heel)*vertical_ref
                left_shoulder_to_right_shoulder = calculate_distance(left_shoulder, right_shoulder)*vertical_ref
                arm_length = calculate_distance(left_wrist, right_wrist)*vertical_ref

                # Display measurements in CMs
                cv2.putText(frame, f'Head to Heel: {head_to_heel:.2f} CMs', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Shoulder to Waist: {shoulder_to_waist:.2f} CMs', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Waist to Heel: {waist_to_heel:.2f} CMs', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Left Shoulder to Right Shoulder: {left_shoulder_to_right_shoulder:.2f} CMs', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Arm Length: {arm_length:.2f} CMs', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Body Pose Landmarks', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
font = cv2.FONT_HERSHEY_PLAIN

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_actual_width = 10  # Change this value to the actual width of your ArUco marker
aruco_actual_height = 10  # Change this value to the actual height of your ArUco marker


def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_marker_dimensions(corners):
    if len(corners) != 4:
        raise ValueError("Four corners are required to calculate dimensions")

    # Extract coordinates of each corner
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    #x4, y4 = corners[3]
    print("corners: ",corners)
    # Calculate distances using Euclidean formula
    width = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) /100
    height = math.sqrt((x3 - x1)**2 + (y3 - y1)**2) /100
    print("width: ",width,"height: ",height)

    vertical_ref=181/height        #change this if doesnt work
    horizontal_ref=182.5/width

    return vertical_ref,horizontal_ref

def main():
    
    size_ranges = {
    'S': (0, 30),
    'M': (31, 49),
    'L': (50, 80),
    'XL': (81, 100),
    'XXL': (101, float('inf')),  # You can adjust the upper limit for XXL as needed
    }
    cap = cv2.VideoCapture(0)  # You can change this to the video file path if you want to process a video.
    vertical_ref=1      # 205        #
    horizontal_ref=1        #205      # change this wrt aruco
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        while cap.isOpened():
            
            ret, rgb_frame = cap.read()
            if not ret:
                
                break
            print("here")
            # Convert the BGR image to RGB
            #rgb_frame = frame   #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                corners, marker_ids, rejected = cv2.aruco.detectMarkers(rgb_frame, dictionary)
            except Exception as e:
                print(e)
                
            print("here2")
            if corners:
                
                for corner, marker_id in zip(corners, marker_ids):
                    # Draw the marker corners.
                    cv2.polylines(
                        rgb_frame, [corner.astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA
                    )

                    # Get the top-right, top-left, bottom-right, and bottom-left corners of the marker.
                    corner = corner.reshape(4, 2)
                    corner = corner.astype(int)

                    top_right, top_left, bottom_right, bottom_left = corner

                    # Write the marker ID on the frame.
                    cv2.putText(
                        rgb_frame, f"id: {marker_id[0]}", top_right, font, 1.3, (255, 0, 255), 2
                    )

                    # Calculate and display the distance of ArUco width and height
                    distance_width, distance_height = calculate_marker_dimensions(corner)
                    
                    cv2.putText(rgb_frame,f"Width: {distance_width:.0f} m, Height: {distance_height:.0f} m",(top_left[0], top_left[1] - 20),font,2.0,(0, 0, 0),2,)
            
                    #vertical_ref=18.1/distance_height
                    #horizontal_ref=18.25/distance_width
                    print("------------------",distance_width, distance_height)
                    vertical_ref,horizontal_ref=calculate_marker_dimensions(corner)
            
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



                print("vertical_ref: ",vertical_ref,"horizontal_ref: ",horizontal_ref)
                # Calculate distances
                head_to_heel = calculate_distance(head, heel)*vertical_ref
                shoulder_to_waist = calculate_distance(shoulder, waist)*vertical_ref
                waist_to_heel = calculate_distance(waist, heel)*vertical_ref
                left_shoulder_to_right_shoulder = calculate_distance(left_shoulder, right_shoulder)*horizontal_ref
                arm_length = calculate_distance(left_wrist, right_wrist)*horizontal_ref

                size_label = None
                for size, (lower, upper) in size_ranges.items():
                    if lower <= shoulder_to_waist <= upper:
                        size_label = size
            
                # Display measurements in CMs
                cv2.putText(rgb_frame, f'Head to Heel: {head_to_heel:.2f} CMs : ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(rgb_frame, f'Shoulder to Waist: {shoulder_to_waist:.2f} CMs -->Your Size: {size_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(rgb_frame, f'Waist to Heel: {waist_to_heel:.2f} CMs', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(rgb_frame, f'Left Shoulder to Right Shoulder: {left_shoulder_to_right_shoulder:.2f} CMs', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(rgb_frame, f'Arm Length: {arm_length:.2f} CMs', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            #cv2.imshow('Body Pose Landmarks', frame)
            cv2.imshow('Aruco', rgb_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

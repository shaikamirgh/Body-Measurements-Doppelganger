import cv2
import numpy as np

# define the fonts for draw text on image
font = cv2.FONT_HERSHEY_PLAIN

# create the dictionary for markers type
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

cap = cv2.VideoCapture(0)

while cap.isOpened:
    ret, image = cap.read()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect ArUco markers in the image.
    corners, marker_ids, rejected = cv2.aruco.detectMarkers(image, dictionary)

    # If markers are detected, draw them on the image.
    if corners:
        # looping through detected markers and marker ids at same time.
        for corner, marker_id in zip(corners, marker_ids):
            # Draw the marker corners.
            cv2.polylines(
                image, [corner.astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA
            )

            # Get the top-right, top-left, bottom-right, and bottom-left corners of the marker.
            # change the shape of numpy array to 4 by 2
            corner = corner.reshape(4, 2)

            # change the type of numpy array values integers
            corner = corner.astype(int)

            # extracting the corner of marker
            top_right, top_left, bottom_right, bottom_left = corner

            # Write the marker ID on the image.
            cv2.putText(
                image, f"id: {marker_id[0]}", top_right, font, 1.3, (255, 0, 255), 2
            )

    # Save the image.
    #cv2.imwrite("out_image1.png", image)
    # Show the image.
    cv2.imshow("image", image)
    # wait until any key press on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cv2.waitKey(0)
# Close all windows.
cv2.destroyAllWindows() 
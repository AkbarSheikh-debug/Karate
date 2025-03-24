from cvzone.PoseModule import PoseDetector
import cv2
import time

# Initialize the PoseDetector
detector = PoseDetector()

# Load the video file
cap = cv2.VideoCapture(0)

gesture = ''
left_punch_count = 0
right_punch_count = 0
left_kick_count = 0
right_kick_count = 0

# Track the last count time
last_count_time = time.time()

while True:
    # Read a frame from the video
    success, img = cap.read()

    # Break the loop if the video has ended
    if not success:
        print("Video ended or failed to load.")
        break

    # Detect pose in the frame
    img = detector.findPose(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, draw=False)

    current_time = time.time()

    if bboxInfo and (current_time - last_count_time >= 1):
        # Define landmarks for the left and right arms
        p1L, p2L, p3L = lmList[11][0:2], lmList[13][0:2], lmList[15][0:2]
        p1R, p2R, p3R = lmList[12][0:2], lmList[14][0:2], lmList[16][0:2]
        p4L, p5L = lmList[23][0:2], lmList[25][0:2]
        p4R, p5R = lmList[24][0:2], lmList[26][0:2]

        # Calculate the angles for the left and right arms
        angArmL, img = detector.findAngle(p1L, p2L, p3L, img=img)
        angArmR, img = detector.findAngle(p1R, p2R, p3R, img=img)
        angLegL, img = detector.findAngle(p1L, p4L, p5L, img=img)
        angLegR, img = detector.findAngle(p1R, p4R, p5R, img=img)

        crossDistL, img, _ = detector.findDistance(lmList[11][0:2], lmList[25][0:2], img)
        crossDistR, img, _ = detector.findDistance(lmList[12][0:2], lmList[26][0:2], img)



        # Detect gestures and update counters
        if detector.angleCheck(angArmL, 80, 30):
            gesture = 'Right Punch'
            right_punch_count += 1
            last_count_time = current_time
        elif detector.angleCheck(angArmR, 270, 40):
            gesture = 'Left Punch'
            left_punch_count += 1
            last_count_time = current_time
        elif crossDistL and crossDistL < 160:
            gesture = 'Right Kick'
            right_kick_count += 1
            last_count_time = current_time
        elif crossDistR and crossDistR < 160:
            gesture = 'Left Kick'
            left_kick_count += 1
            last_count_time = current_time

        # Display the detected gesture and counts on the frame
        cv2.putText(img, f"Gesture: {gesture}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        cv2.putText(img, f"Left Punches: {left_punch_count}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Right Punches: {right_punch_count}", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Left Kicks: {left_kick_count}", (20, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.putText(img, f"Right Kicks: {right_kick_count}", (20, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Pose Detection", img)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

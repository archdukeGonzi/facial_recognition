import threading
from threading import Thread

import cv2
from deepface import DeepFace

# Global variables
face_match = False
model_loaded = True
frame_to_process = None
counter = 0

# Reference image for facial recognition
reference_img = cv2.imread("Photo on 8-25-24 at 11.48 AM.jpg")


# initialize video capture
cap = cv2.VideoCapture(0)

# setting properties of window
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

# error handling if camera cannot open
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


# Load the facial recognition model once before the loop that does the work
try:
    DeepFace.verify(reference_img, reference_img)  # dummy call to load the model
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False



# function to process face recognition in a background thread
def check_face():
    global face_match, frame_to_process
    while True:
        if frame_to_process is not None:
            try:
                face_match = DeepFace.verify(frame_to_process, reference_img, model_name="Facenet")['verified']
            except Exception as e:
                print(f"Error during face verification: {e}")
                face_match = False
            # reset the frame_to_process to avoid repeated processing
            frame_to_process = None

threading.Thread(target=check_face, daemon=True).start()

while model_loaded:
    ret, frame = cap.read()
    if ret:
        if counter % 60 == 0:  # Check every 30 frames
            frame_to_process = frame.copy()

        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Facial Recognition", frame)
    else:
        print("Failed to capture frame")
        break

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()


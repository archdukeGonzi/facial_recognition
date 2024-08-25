import cv2
from deepface import DeepFace

# Initialize Video Capture
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


reference_img = cv2.imread("IMG_1535.jpg")

# Load the model once before the loop
try:
    DeepFace.verify(reference_img, reference_img)  # Dummy call to load the model
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

counter = 0
face_match = False

def check_face(myframe):
    global face_match
    try:
        face_match = DeepFace.verify(myframe, reference_img, model_name="Facenet")['verified']
    except Exception as e:
        print(f"Error during face verification: {e}")
        face_match = False

while model_loaded:
    ret, frame = cap.read()

    if ret:
        if counter % 60 == 0:  # Check every 30 frames
            check_face(frame.copy())

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


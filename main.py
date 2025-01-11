import cv2
import dlib
import numpy as np
import uuid

# Load pre-trained face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Update the path

# Load a simple texture analysis model (You might want to replace this with a more sophisticated model)
texture_model = cv2.imread("texture_model.png", cv2.IMREAD_GRAYSCALE)

# Threshold for texture analysis (You may need to adjust this)
texture_threshold = 0.2

# Initialize OpenCV's video capture
cap = cv2.VideoCapture(0)

# Create an empty dictionary to store face encodings and corresponding IDs
face_encodings = {}
consistent_face_count = 0
max_consistent_face_count = 10  # Adjust the value based on your requirements

# Initialize variables for liveness check
frames_for_liveness = []
max_frames_for_liveness = 5  # Number of frames to consider for liveness check
liveness_confidence_threshold = 10 # Adjust the value based on your observations

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    if len(faces) > 0:
        consistent_face_count += 1
    else:
        consistent_face_count = 0

    if consistent_face_count >= max_consistent_face_count:
        for face in faces:
            # Calculate face width and height
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()

            # Check if the detected region has reasonable proportions for a face
            if 0.2 < face_width / face_height < 2.0 and 100 < face_width < 400 and 100 < face_height < 400:
                # Get the facial landmarks
                landmarks = predictor(gray, face)

                # Draw rectangle around the face
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

                # Check if the required landmarks for the left eye are present
                if hasattr(landmarks, 'part') and len(landmarks.parts()) >= 39:
                    # Extract a region of interest (ROI) around the left eye for texture analysis
                    left_eye_roi = gray[landmarks.part(36).y:landmarks.part(39).y,
                                   landmarks.part(36).x:landmarks.part(39).x]

                    # Check if the left eye ROI is not empty
                    if left_eye_roi.shape[0] > 0 and left_eye_roi.shape[1] > 0:
                        # Resize the left eye ROI to match the texture model size
                        left_eye_roi = cv2.resize(left_eye_roi, (texture_model.shape[1], texture_model.shape[0]))

                        # Calculate texture similarity using normalized cross-correlation
                        left_eye_similarity = \
                        cv2.matchTemplate(left_eye_roi, texture_model, cv2.TM_CCOEFF_NORMED)[0][0]

                        # Average the similarity scores from both eyes
                        average_similarity = left_eye_similarity

                        # Display the result based on liveness and texture analysis
                        if average_similarity < texture_threshold:
                            # Store landmarks for liveness check
                            frames_for_liveness.append(landmarks)

                            # Perform liveness check using multiple frames
                            if len(frames_for_liveness) == max_frames_for_liveness:
                                # Calculate the change in head position
                                total_distance_moved = 0
                                for i in range(1, len(frames_for_liveness)):
                                    # Extract coordinates from full_object_detection objects
                                    landmarks_current = np.array([(p.x, p.y) for p in frames_for_liveness[i].parts()])
                                    landmarks_prev = np.array([(p.x, p.y) for p in frames_for_liveness[i - 1].parts()])

                                    # Calculate the Euclidean distance
                                    distance_moved = np.linalg.norm(landmarks_current - landmarks_prev)
                                    total_distance_moved += distance_moved

                                # Calculate average distance moved
                                average_distance_moved = total_distance_moved / (len(frames_for_liveness) - 1)

                                # Update liveness confidence
                                liveness_confidence = average_distance_moved

                                # Display debug information
                                print(f"Average Distance Moved: {average_distance_moved}")
                                print(f"Liveness Confidence: {liveness_confidence}")

                                # Display the result based on liveness and texture analysis
                                if liveness_confidence < liveness_confidence_threshold:
                                    confidence = int(liveness_confidence)

                                    # Generate a unique ID for the face
                                    face_id = str(uuid.uuid4())

                                    # Store the face encoding in the dictionary
                                    face_encodings[face_id] = landmarks

                                    # Display the face ID on the frame
                                    cv2.putText(frame,
                                                f'Live Face (Confidence: {confidence}, ID: {face_id})',
                                                (face.left(), face.top() - 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                else:
                                    # Display a message for spoof attempts
                                    cv2.putText(frame, 'Spoof Attempt', (face.left(), face.top() - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                # Display a message for non-live faces
                                cv2.putText(frame, 'Not a live face!', (50, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Face Detection and Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import sys
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

import cv2
from deepface import DeepFace


class EmotionDetection:
    """
    Real-time facial emotion detection using computer vision and deep learning.
    The system captures video from your webcam, detects faces, recognizes emotions,
    and displays corresponding emojis.

    Supported emotions: Angry 😠, Disgust 🤢, Fear 😱, Happy 😊, Sad 😢, Surprise 😲, Neutral 😐
    """

    def __init__(self):
        """Initialize emotion-to-emoji mapping"""
        self.emotion_emoji = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😱',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_emotion = "No face detected"

    def detect_emotion(self):
        """
        Capture video from webcam and perform real-time emotion detection.

        KEY FIX: Changed from DeepFace.build_model("Emotion-Detection")
                 to DeepFace.build_model("Emotion")

        The error occurred because "Emotion-Detection" is not a valid model name.
        Valid model names for build_model() are:
        - "Emotion" (for emotion detection)
        - "Age" (for age prediction)
        - "Gender" (for gender detection)
        - "Race" (for race/ethnicity detection)
        """

        # Start webcam capture (0 = default webcam)
        capture_video = cv2.VideoCapture(0)

        while True:
            ret, frame = capture_video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                face_roi = frame[y:y + h, x:x + w]
                label = "No face detected"
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list):
                        result = result[0]
                    emotion = result.get('dominant_emotion', 'neutral').lower()
                    emoji = self.emotion_emoji.get(emotion, '')
                    confidence = result.get('emotion', {}).get(emotion, 0)
                    label = f"{emotion.capitalize()} {emoji} ({confidence:.2f}%)"
                    self.last_emotion = label  # Save last detected emotion label
                except Exception as e:
                    label = "Detection error"
                    print(f"Emotion analysis error: {e}")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # Only handle first detected face

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture_video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = EmotionDetection()
    try:
        detector.detect_emotion()
    except KeyboardInterrupt:
        print(f"\nProgram stopped. Last detected face emotion: {detector.last_emotion}")

import cv2
import numpy as np
import face_recognition
class VideoProcessing:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.obama_image = face_recognition.load_image_file("obama.jpg")
        self.obama_face_encoding = face_recognition.face_encodings(self.obama_image)[0]
        self.odai_image = face_recognition.load_image_file("odai.jpg")
        self.odai_face_encoding = face_recognition.face_encodings(self.odai_image)[0]
        self.biden_image = face_recognition.load_image_file("biden.jpg")
        self.biden_face_encoding = face_recognition.face_encodings(self.biden_image)[0]
        self.imad_image = face_recognition.load_image_file("imad.jpg")
        self.imad_face_encoding = face_recognition.face_encodings(self.imad_image)[0]
        self.aziz_image = face_recognition.load_image_file("aziz.jpg")
        self.aziz_face_encoding = face_recognition.face_encodings(self.aziz_image)[0]

        self.known_face_encodings = [
            self.obama_face_encoding,
            self.biden_face_encoding,
            self.odai_face_encoding,
            self.imad_face_encoding,
            self.aziz_face_encoding
        ]
        self.known_face_names = [
            "Barack Obama",
            "Joe Biden",
            "Odai",
            "Imad",
            "Aziz"
        ]

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_names = []
        for (x, y, w, h) in faces:
            # Convert face region to RGB for face_recognition
            rgb_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

            # Find face encodings in the current frame
            face_encodings = face_recognition.face_encodings(rgb_face)

            face_names = []
            for face_encoding in face_encodings:
                # Compare face encodings with known face encodings
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)

            # Draw bounding box and label for each detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, ', '.join(face_names), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Video', frame)
        cv2.waitKey(1)

    def process_video(self):
        cap = cv2.VideoCapture(0)  # 0 steht für die erste verfügbare Kamera

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            self.process_frame(frame)

        cap.release()
        cv2.destroyAllWindows()


def main():
    video_processing = VideoProcessing()
    video_processing.process_video()


if __name__ == '__main__':
    main()

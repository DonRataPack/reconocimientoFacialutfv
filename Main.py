import cv2
import time

# Cargar el clasificador en cascada
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

screenshot_counter = 0
max_screenshots_per_face = 3
previous_face_location = None
screenshot_taken_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Verificar si es un rostro nuevo comparando su ubicación con la anterior
        if previous_face_location is None or (abs(x - previous_face_location[0]) > 10 or
                                              abs(y - previous_face_location[1]) > 10):
            screenshot_counter = 0  # Reiniciar el contador para un nuevo rostro
            previous_face_location = (x, y)  # Actualizar la ubicación

        # Tomar las capturas si aún no hemos llegado al límite
        if screenshot_counter < max_screenshots_per_face:
            roi_color = frame[y:y + h, x:x + w]
            filename = f"face_screenshot_{time.strftime('%Y%m%d_%H%M%S')}_{screenshot_counter}.jpg"
            cv2.imwrite(filename, roi_color)
            screenshot_counter += 1

    cv2.imshow('Detección de Rostros', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

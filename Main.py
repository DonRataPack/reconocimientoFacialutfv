import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

screenshot_counter = 0
max_screenshots_per_face = 3
previous_face_location = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 5.1, 5, 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            # Verificamos si es un rostro nuevo comparando su ubicación con el anterior
            if previous_face_location is None or (abs(x - previous_face_location[0]) > 10 or
                                                  abs(y - previous_face_location[1]) > 10):
                screenshot_counter = 0  # Reiniciamos el contador para el nuevo rostro
                previous_face_location = (x, y)  # Actualizamos la ubicación

            # Tomamos las capturas si aún no hemos llegado al límite
            if screenshot_counter < max_screenshots_per_face:
                roi_color = frame[y:y + h, x:x + w]
                cv2.imwrite(f"face_screenshot_{screenshot_counter}.jpg", roi_color)
                screenshot_counter += 1
            screenshot_counter = 0

    cv2.imshow('Deteccion de Rostros', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf

# Загрузка tflite модели
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Получение входного/выходного тензора
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Размер, как в обучении
width, height = 160, 120

# Подключение камеры
cap = cv2.VideoCapture(0)

label_map = {0: 'Left', 1: 'Right'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (width, height))
    normalized = resized / 255.0
    input_data = np.array(normalized, dtype=np.float32).reshape(1, width, height)

    # Предсказание
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output)

    direction = label_map[prediction]

    # Отображение текста на видео
    cv2.putText(frame, f'Direction: {direction}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Direction Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

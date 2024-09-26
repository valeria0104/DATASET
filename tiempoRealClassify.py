import cv2
import SeguimientoManos as sm
from ultralytics import YOLO

# Cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Ajustar resolución de la cámara
cap.set(4, 720)

# Cargar modelo de clasificación YOLO
model = YOLO('best.pt')

# Detector de manos
detector = sm.detectarmanos(Confdeteccion=0.9)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen de la cámara.")
        break

    # Detectar las manos en el frame
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Obtener la posición de la primera mano detectada
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])

    if mano == 1:
        # Extraer las coordenadas del bounding box (cuadro delimitador)
        xmin, ymin, xmax, ymax = bbox
        
        # Asignar un margen alrededor de la mano
        xmin = max(0, xmin - 40)
        ymin = max(0, ymin - 40)
        xmax = min(frame.shape[1], xmax + 60)  # frame.shape[1] es el ancho de la imagen
        ymax = min(frame.shape[0], ymax + 40)  # frame.shape[0] es el alto de la imagen

        # Verificar que el recorte es válido
        if xmax > xmin and ymax > ymin:
            # Recortar la región de la mano
            recorte = frame[ymin:ymax, xmin:xmax]

            # Redimensionar la imagen recortada al tamaño esperado por el modelo YOLO
            recorte = cv2.resize(recorte, (416, 416), interpolation=cv2.INTER_CUBIC)

            # Hacer predicción con el modelo de clasificación YOLO
            resultados = model.predict(recorte, conf=0.55)

            if resultados:
                for result in resultados:
                    if result.probs is not None:
                        # Obtener la clase con mayor probabilidad
                        clase_id = result.probs.top1  # Índice de la clase con mayor probabilidad
                        clase_nombre = result.names[clase_id]  # Obtener el nombre de la clase
                        conf = result.probs.top1conf  # Obtener la confianza asociada

                        # Mostrar el nombre de la clase y la confianza en el frame
                        cv2.putText(frame, f'Clase: {clase_nombre} ({conf:.2f})', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        else:
            print("Coordenadas de recorte no válidas:", xmin, ymin, xmax, ymax)

    # Mostrar el frame principal con las predicciones
    cv2.imshow("Lenguaje abecedario", frame)

    # Salir si se presiona ESC
    t = cv2.waitKey(1)
    if t == 27:
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

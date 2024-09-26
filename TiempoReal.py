import cv2
import SeguimientoManos as sm
from ultralytics import YOLO

# cámara
cap = cv2.VideoCapture(0)
# cambiando resolución
cap.set(3, 1280)
cap.set(4, 720)

# Cargar modelo
model = YOLO('bestAbel.pt')  # Cargamos el modelo correctamente usando la clase YOLO

# detector
detector = sm.detectarmanos(Confdeteccion=0.9)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen de la cámara.")
        break

    # Extraer info de la mano
    frame = detector.encontrarmanos(frame, dibujar=False)

    # Posición de una mano
    lista1, bbox, mano = detector.encontrarposicion(frame, ManoNum=0, dibujarPuntos=False, dibujarBox=False, color=[0, 255, 0])

    if mano == 1:
        # Extraer la información del cuadro
        xmin, ymin, xmax, ymax = bbox
        # Asignar margen
        xmin = max(0, xmin - 40)
        ymin = max(0, ymin - 40)
        xmax = min(frame.shape[1], xmax + 60)  # frame.shape[1] es el ancho de la imagen
        ymax = min(frame.shape[0], ymax + 40)  # frame.shape[0] es el alto de la imagen

        # Asegurarse de que el recorte es válido
        if xmax > xmin and ymax > ymin:
            # Recorte de mano
            recorte = frame[ymin:ymax, xmin:xmax]

            # Redimensionar
            recorte = cv2.resize(recorte, (416, 416), interpolation=cv2.INTER_CUBIC)

            # Extraer resultados
            resultados = model.predict(recorte, conf=0.55)
            
            if len(resultados) != 0:
                for result in resultados:
                    masks = result.masks
                    coordenadas = masks

                    anotaciones = resultados[0].plot()

            cv2.imshow("RECORTE", anotaciones)
        else:
            print("Coordenadas de recorte no válidas:", xmin, ymin, xmax, ymax)
    
    cv2.imshow("Lenguaje abecedario", frame)

    # Leer teclado
    t = cv2.waitKey(1)
    if t == 27:  # Si presionas ESC, rompe el ciclo
        break

cap.release()
cv2.destroyAllWindows()

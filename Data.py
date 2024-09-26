import os 
import cv2
import SeguimientoManos as sm

#creando carpetas
nombre = 'Y'
direccion = 'C:/Users/VALERIA/Downloads/LetrasFinal/Letras'
carpeta = direccion + '/' + nombre 
#si no esta creada la carpeta
if not os.path.exists(carpeta):#la carpeta esta creada
    print("CARPETA CREADA: ",carpeta)
    os.makedirs(carpeta)

#camara
cap = cv2.VideoCapture(0)
#cambiando resolucion
cap.set(3,1280)
cap.set(4,720)

cont = 80
#detector
detector = sm.detectarmanos(Confdeteccion= 0.9)

while True:
    ret, frame  = cap.read()
    #extraer info de la mano
    frame = detector.encontrarmanos(frame, dibujar=False)

    # posicion de una mano
    lista1, bbox, mano = detector.encontrarposicion(frame,ManoNum= 0, dibujarPuntos= False, dibujarBox= False, color= [0,255,0])

    if mano == 1:
        #Extraer la informacion del cuadro 

        xmin,ymin,xmax,ymax = bbox
        #asignar margen
        xmin = xmin -40
        ymin = ymin -40
        xmax = xmax +60
        ymax = ymax + 40
        #recorte mano
        recorte = frame[ymin:ymax,xmin:xmax]
        
        #redimensionar
        recorte = cv2.resize(recorte, (416,416), interpolation= cv2.INTER_CUBIC)

        #almacenar imagenes
        cv2.imwrite(carpeta + "/Y_{}.jpg".format(cont), recorte)
        
        #aumentar cont
        cont = cont +1 
        cv2.imshow("RECORTE", recorte)
        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0,255,0],2)
    
    cv2.imshow("Lenguaje abecedario", frame)
    #leer teclado
    t = cv2.waitKey(1)
    if t == 27 or cont == 120:
        break

cap.release()
cv2.destroyAllWindows()
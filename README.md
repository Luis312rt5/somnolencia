Detector de somnolencia v3.1  -  Raspberry Pi 4 + camara USB + buzzer
 
Que hace:
  * Rostro: malla de 478 puntos (MediaPipe Face Landmarker). Con los puntos de los
    parpados calcula el EAR (apertura del ojo), calibrado para cada persona.
  * Cabeza: calcula giro e inclinacion; avisa si la cabeza se desvia o cae.
  * Manos: 21 puntos por mano (MediaPipe Hand Landmarker). Si no hay manos dentro de
    la zona del volante, suena la alarma.
  * Alcance: busca el rostro por mosaicos y luego lo sigue con un recorte alrededor
    de la cara, asi funciona con la camara lejos.
  * FPS: la camara se lee en un hilo aparte (siempre el cuadro mas reciente, sin
    retraso) y las manos se procesan en otro hilo.
  * Solo el conductor: el rostro se busca unicamente dentro de la ZONA DEL CONDUCTOR,
    se descartan caras lejanas a la que se sigue, y se ignoran caras/manos pequenas
    (los pasajeros de atras se ven mas pequenos).
 
Teclas (con ventana):  q = salir | c = recalibrar ojos/cabeza
                       k = marcar zona del conductor (cabeza) | v = marcar zona del volante (manos)
Opciones:  --info-camara  (muestra zoom y resoluciones de la camara y sale)
           --sin-ventana  --sin-manos  --sin-calibrar  --camara N
           --guardar-img  (guarda /tmp/vista.jpg con los dibujos, para verla por SSH)

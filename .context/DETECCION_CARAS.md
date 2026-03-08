Es normal que la detección de caras falle en un partido de fútbol: las caras son muy pequeñas (poca resolución), hay desenfoque por movimiento y los ángulos son picados. Los algoritmos estándar están entrenados para "selfies" o gente cerca de la cámara.

Para tu RTX 5070 Ti (arquitectura Blackwell), tienes potencia para usar modelos mucho más agresivos que un simple clasificador. Aquí tienes la estrategia para 2026:

1. El Algoritmo: Olvida MediaPipe, ve a por YOLOv12 o SCRFD
Si las caras son pequeñas y están lejos, necesitas modelos con "Attention" o específicos para escalas pequeñas:

YOLOv12-Face (Recomendado): Es la novedad de 2026. Utiliza Area Attention, lo que le permite entender el contexto (un cuerpo de futbolista) para saber dónde debería estar la cara aunque no se vea perfecta. Es mucho más preciso para objetos lejanos que las versiones anteriores.

SCRFD (Sample and Computation Redistribution for Face Detection): Si buscas la máxima precisión en caras minúsculas. Es el estándar actual en InsightFace para vigilancia y deportes. Detecta caras de apenas 10x10 píxeles.

RetinaFace (ResNet-50): Un clásico que sigue siendo imbatible en robustez si no te importa que consuma un poco más de VRAM.

2. El Secreto: Detección + Tracking (ByteTrack)
El "flickering" (que la cara aparezca y desaparezca) no se arregla solo con un mejor detector, sino con un Tracker.
En lugar de detectar en cada frame desde cero:

Detectas en el Frame 1.

Usas ByteTrack o OC-SORT para "predecir" dónde estará la cara en el Frame 2.

Si el detector falla en un frame, el Tracker mantiene la posición de la máscara basándose en la inercia del movimiento del jugador.

3. Implementación con PyTorch y tu RTX 5070 Ti
Al ser una tarjeta Blackwell, debes usar PyTorch Nightly y TensorRT para aprovechar los núcleos Tensor de 5ª generación.

Estructura de código sugerida:
Python
import torch
from ultralytics import YOLO # Para YOLOv12

# 1. Cargar modelo optimizado para caras
model = YOLO('yolov12n-face.pt').to('cuda')

# 2. Si quieres usar la potencia real de la 5070 Ti, exporta a TensorRT
# model.export(format='engine', dynamic=True, half=True)

def process_frame(frame):
    # Detección con umbral bajo para no perder caras lejanas
    results = model.predict(frame, conf=0.25, iou=0.45, verbose=False)
    
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int()
            
            # SUSTITUCIÓN POR DIBUJO
            # En lugar de solo borrar, haz un 'overlay' de un PNG de cara cartoon
            # ajustado al tamaño de la caja (x2-x1)
            frame = apply_cartoon_mask(frame, (x1, y1, x2, y2))
            
    return frame
4. Mejorar la "Máscara de Dibujo"
Si quieres que la cara sea un dibujo animado consistente, no la borres. Usa Landmarks (puntos faciales).

Incluso si la cara está lejos, modelos como YOLOv12-Face te dan 5 puntos clave (ojos, nariz, comisuras).

Usa esos 5 puntos para rotar y escalar tu dibujo animado. Así, si el jugador inclina la cabeza, el dibujo también lo hará.

5. Recomendación de "Skill" de Vídeo
Busca la librería SAHI (Slicing Aided Hyper Inference).

Cómo funciona: Divide el frame del campo de fútbol en 4 u 8 trozos, detecta caras en cada trozo a alta resolución y luego junta los resultados.

Por qué para ti: Con tu GPU, puedes procesar esos trozos en paralelo. Esto hará que detectes caras que para un algoritmo normal son simples "manchas".

¿Cuál es el siguiente paso?
Para que el proceso sea fluido, el cuello de botella será la transferencia de datos entre CPU y GPU. ¿Te gustaría que te ayude a configurar un pipeline con CuPy o PyTorch para que el procesamiento de la máscara se haga enteramente en la VRAM sin volver a la RAM del sistema? Con una 5070 Ti, esto te permitiría subir de 15fps a 60fps fácilmente.
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Función principal
def capture_iris_image():
    cap = cv.VideoCapture(0)
    captured_image = None
    eye_detected_time = None
    frame_with_contours = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                
                frame_with_contours = frame.copy()
                cv.circle(frame_with_contours, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame_with_contours, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

                if eye_detected_time is None:
                    eye_detected_time = time.time()  # Inicia el temporizador
                elif time.time() - eye_detected_time >= 5:
                    captured_image = frame.copy()  # Captura la imagen sin contornos
                    break
            else:
                eye_detected_time = None  # Reinicia el temporizador si no se detectan ojos
            
            # Muestra el frame con contornos si están disponibles, de lo contrario muestra el frame original
            if frame_with_contours is not None:
                cv.imshow('img', frame_with_contours)
            else:
                cv.imshow('img', frame)
            
            key = cv.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()
    return captured_image

# Captura la imagen del iris
captured_image = capture_iris_image()

if captured_image is not None:
    # Convierte la imagen capturada de BGR a RGB
    captured_image_rgb = cv.cvtColor(captured_image, cv.COLOR_BGR2RGB)
    
    # Guarda la imagen como archivo
    cv.imwrite('captured_iris_image.jpg', captured_image_rgb)
    print("Imagen capturada y guardada exitosamente.")
    
    # Plotea la imagen usando matplotlib
    plt.imshow(captured_image_rgb)
    plt.axis('off')  # Oculta los ejes
    plt.show()
else:
    print("No se capturó ninguna imagen.")

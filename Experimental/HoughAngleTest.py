import cv2
import numpy as np

captura = cv2.VideoCapture("/home/max/Documents/Ternium-Tests/Database/testCam/WoodTest0_100cm_123_.avi")

if not captura.isOpened():
    print("Erro ao acessar a câmera ou o arquivo de vídeo.")
    exit()

cv2.namedWindow('Detecção de Empeno em Vídeo', cv2.WINDOW_NORMAL)

while True:
    ret, frame = captura.read()
    
    if not ret:
        print("Fim do vídeo ou erro de leitura.")
        break

    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    bordas = cv2.Canny(imagem_suavizada, 50, 150)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    empeno_detectado = False

    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            angulo = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            if 2 < abs(angulo) < 10 :
                empeno_detectado = True
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                print(f'Empeno detectado com ângulo: {angulo:.2f} graus')
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Detecção de Empeno em Vídeo', frame)

    if empeno_detectado:
        print("Empeno detectado! Vídeo pausado. Pressione 'c' para continuar.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('c'):
                break
            elif key == ord('q'):
                captura.release()
                cv2.destroyAllWindows()
                exit()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()

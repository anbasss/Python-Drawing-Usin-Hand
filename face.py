import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variabel untuk menyimpan posisi gambar
drawing_points = []

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]  # ID landmark untuk ujung jari
    thumb_tip = 4
    fingers_up = []

    # Periksa setiap jari (kecuali ibu jari)
    for tip_id in finger_tips[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    # Periksa ibu jari
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 2].x:
        fingers_up.insert(0, 1)
    else:
        fingers_up.insert(0, 0)

    return sum(fingers_up)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Balik gambar secara horizontal
    frame = cv2.flip(frame, 1)

    # Konversi gambar ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Menggambar landmark dan koneksi di tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Menghitung jumlah jari yang terangkat
            fingers_count = count_fingers(hand_landmarks)

            # Jika lima jari dibuka, hapus semua coretan
            if fingers_count == 5:
                drawing_points = []
            else:
                # Dapatkan posisi jari telunjuk
                index_finger_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                drawing_points.append((x, y))

    # Menggambar pada frame
    for point in drawing_points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)

    # Menampilkan frame
    cv2.imshow("Hand Gesture Drawing", frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clear All Exp 
cap.release()
cv2.destroyAllWindows()

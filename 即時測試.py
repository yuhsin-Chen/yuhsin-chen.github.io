import cv2
import mediapipe as mp
import numpy as np
import joblib
import sys

# 載入模型與標籤轉換器
try:
    model = joblib.load("asl_random_forest_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ 模型與標籤轉換器載入成功")
except Exception as e:
    print("❌ 模型載入失敗：", e)
    sys.exit(1)

# 初始化 MediaPipe 手部偵測
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 開啟攝影機
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ 無法開啟攝影機")
    sys.exit(1)

print("📷 開始進行即時手語辨識，按下 q 鍵可退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 攝影機畫面讀取失敗")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 擷取相對座標
            landmarks = []
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z - base_z
                ])

            if len(landmarks) != 63:
                continue

            try:
                probs = model.predict_proba([landmarks])[0]
                top3_idx = np.argsort(probs)[-3:][::-1]
                top3 = [(label_encoder.inverse_transform([i])[0], probs[i]) for i in top3_idx]

                # 顯示主要預測
                main_pred, main_prob = top3[0]
                cv2.putText(frame, f'{main_pred} ({main_prob:.2f})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 200), 3)

                # 顯示 Top-3
                y = 80
                for label, prob in top3:
                    cv2.putText(frame, f'{label}: {prob:.2f}', (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    y += 30

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            except Exception as e:
                print("❌ 預測錯誤：", e)

    try:
        cv2.imshow("ASL Hand Translate", frame)
    except Exception as e:
        print("❌ 顯示畫面錯誤：", e)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 已退出手語辨識")
        break

# 清理
cap.release()
cv2.destroyAllWindows()

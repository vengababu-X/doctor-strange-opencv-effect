import cv2
import mediapipe as mp
import math
import os

# --- CONFIGURATION ---
image_name = '1.png.png'

# Check if image is found
if not os.path.exists(image_name):
    print(f"Error: I cannot find {image_name}. Make sure it is in the same folder!")
    exit()

# --- SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Load the shield image
shield_img = cv2.imread(image_name, -1)

def overlay_transparent(background, overlay, x, y, size):
    # Resize shield to match hand size
    overlay = cv2.resize(overlay, (size, size))
    h, w, c = overlay.shape
    bg_h, bg_w, _ = background.shape

    # Calculate center position
    x_offset = x - w // 2
    y_offset = y - h // 2

    # Check boundaries (don't crash if hand goes off screen)
    if x_offset < 0: x_offset = 0
    if y_offset < 0: y_offset = 0
    if x_offset + w > bg_w: w = bg_w - x_offset
    if y_offset + h > bg_h: h = bg_h - y_offset
    
    if w <= 0 or h <= 0: return background

    overlay_cropped = overlay[0:h, 0:w]
    
    # Blend the images (Transparency logic)
    alpha_mask = overlay_cropped[:, :, 3] / 255.0
    img_rgb = overlay_cropped[:, :, :3]
    
    roi = background[y_offset:y_offset+h, x_offset:x_offset+w]
    
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - alpha_mask) + img_rgb[:, :, c] * alpha_mask

    background[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    return background

# --- START CAMERA ---
cap = cv2.VideoCapture(0)
deg = 0

print("System Ready. Open your hand!")

while True:
    success, img = cap.read()
    if not success: break
    
    # Flip the image (Mirror effect)
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape
            
            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            if lm_list:
                # 4=Thumb, 20=Pinky, 9=Center Palm
                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[20][1], lm_list[20][2]
                cx, cy = lm_list[9][1], lm_list[9][2]
                
                # Check how wide the hand is open
                hand_size = math.hypot(x2 - x1, y2 - y1)
                
                # IF OPEN -> SHOW SHIELD
                if hand_size > 100:
                    deg += 3
                    if deg >= 360: deg = 0
                    
                    # Rotate the shield
                    rows, cols, _ = shield_img.shape
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
                    rotated_shield = cv2.warpAffine(shield_img, M, (cols, rows))
                    
                    # Scale and Overlay
                    shield_diameter = int(hand_size * 2.8)
                    img = overlay_transparent(img, rotated_shield, cx, cy, shield_diameter)
                    
                # IF CLOSED -> SHOW LINES
                else:
                    mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Doctor Strange Portal", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

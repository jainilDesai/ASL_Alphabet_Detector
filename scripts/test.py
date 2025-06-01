import cv2

print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam.")
    exit()

print("✅ Webcam opened!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Failed to grab frame.")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

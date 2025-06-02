import cv2

# Step 1: Capture image from webcam
cam = cv2.VideoCapture(0)
print("📸 Press SPACE to take a photo")
while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    cv2.imshow("Camera - Press SPACE to capture", frame)

    k = cv2.waitKey(1)
    if k%256 == 32:  # SPACE pressed
        img_name = "test.jpg"
        cv2.imwrite(img_name, frame)
        print("✅ Photo saved as", img_name)
        break

cam.release()
cv2.destroyAllWindows()

# Step 2: Load age detection model
age_model = cv2.dnn.readNetFromCaffe(
    'age_deploy.prototxt',
    'age_net.caffemodel'
)

# Age buckets
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Step 3: Load saved photo and detect age
image = cv2.imread('test.jpg')
if image is None:
    print("no, test.jpg not found!")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    face_img = image[y:y+h, x:x+w]
    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    age_model.setInput(blob)
    predictions = age_model.forward()
    age = AGE_BUCKETS[predictions[0].argmax()]
    
    label = f"Age: {age}"
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 0, 0), 2)

cv2.imshow("Age Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


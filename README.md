# Automated Parking System Using License PlateDetection

Below is a **detailed explanation of the execution flow** of the code, including when each function is called and how it contributes to the overall process. 

---

### **1. Program Starts**
- The program begins execution from the `if __name__ == "__main__":` block.
- The `start_time` is recorded to measure the total execution time.

```python
if __name__ == "__main__":
    start_time = time.time()
    results = main()
    print("------------------------Processing completed.------------")
    print("Time taken: {:.2f} seconds".format(time.time() - start_time))
    print("Results:", results)
    write_csv(results, './results.csv')
```

---

### **2. `main()` Function Execution**
The `main()` function is the core of the program. It orchestrates the entire process.

#### **Step 1: Initialize and Verify Models**
- The `initialize_models()` function is called to load and verify the required models.
  - **`initialize_models()`**:
    - Calls `verify_models()` to ensure all models (YOLOv8, license plate detector, and EasyOCR) are loaded correctly.
    - If successful, it initializes and returns the models.

```python
def initialize_models():
    models = {}
    try:
        success, message = verify_models()
        if not success:
            raise Exception(message)
        
        models['coco_model'] = YOLO('yolov8n.pt')  # COCO model for vehicle detection
        models['license_plate_detector'] = YOLO('license_plate_detector.pt')  # License plate detector
        models['reader'] = easyocr.Reader(['en'], gpu=False)  # EasyOCR for text extraction
        return models
    except Exception as e:
        raise Exception(f"Failed to initialize models: {str(e)}")
```

- **`verify_models()`**:
  - Checks if the model files exist.
  - Tests each model (YOLOv8, license plate detector, and EasyOCR) by running inference on a test image.
  - Returns `True` if all models are verified successfully.

```python
def verify_models():
    try:
        # Verify license plate detector
        if not os.path.exists('license_plate_detector.pt'):
            return False, "License plate detector model file not found"
        
        license_plate_detector = YOLO('license_plate_detector.pt')
        test_image = np.zeros((416, 640, 3), dtype=np.uint8)
        _ = license_plate_detector(test_image)
        print("✓ License plate detector verified successfully")
        
        # Verify COCO model
        coco_model = YOLO('yolov8n.pt')
        _ = coco_model(test_image)
        print("✓ COCO model verified successfully")
        
        # Verify EasyOCR
        reader = easyocr.Reader(['en'], gpu=False)
        test_text_image = np.ones((100, 200), dtype=np.uint8) * 255
        cv2.putText(test_text_image, 'TEST', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        _ = reader.readtext(test_text_image)
        print("✓ EasyOCR model verified successfully")
        
        return True, "All models verified successfully"
    except Exception as e:
        return False, f"Unexpected error during model verification: {str(e)}"
```

---

#### **Step 2: Initialize Video Capture**
- The video file is opened using `cv2.VideoCapture()`.
- If the video file cannot be opened, an exception is raised.

```python
video_path = r'/content/drive/MyDrive/4-2 Project/Automatic-License-Plate-Recognition-using-YOLOv8-main/traffic_4secs.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"Error: Could not open video file at {video_path}")
```

---

#### **Step 3: Initialize Tracker**
- The **SORT tracker** is initialized to track vehicles across frames.

```python
mot_tracker = Sort()
```

---

#### **Step 4: Process Video Frames**
- The program processes each frame of the video in a loop.

##### **a. Detect Vehicles**
- The COCO model (`coco_model`) is used to detect vehicles in the current frame.
- Only vehicles of specific classes (car, motorcycle, bus, truck) are considered.

```python
detections = coco_model(frame)[0]
detections_ = []
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])
```

##### **b. Track Vehicles**
- The SORT tracker is used to track vehicles across frames.
- The `track_ids` variable stores the tracked vehicle IDs and their bounding boxes.

```python
track_ids = np.empty((0, 5))
if len(detections_) > 0:
    track_ids = mot_tracker.update(np.asarray(detections_))
```

##### **c. Detect License Plates**
- The license plate detector (`license_plate_detector`) is used to detect license plates in the current frame.

```python
license_plates = license_plate_detector(frame)[0]
for license_plate in license_plates.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = license_plate
```

##### **d. Associate License Plates with Vehicles**
- The `get_car()` function is called to associate a detected license plate with a tracked vehicle.

```python
xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
```

- **`get_car()`**:
  - Compares the license plate bounding box with the vehicle bounding boxes to find a match.
  - Returns the vehicle ID and bounding box if a match is found.

```python
def get_car(license_plate, vehicle_track_ids):
    try:
        x1, y1, x2, y2, score, class_id = license_plate
        for j in range(len(vehicle_track_ids)):
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return vehicle_track_ids[j]
        return -1, -1, -1, -1, -1
    except Exception as e:
        print(f"Error in get_car: {str(e)}")
        return -1, -1, -1, -1, -1
```

##### **e. Extract License Plate Text**
- The `read_license_plate()` function is called to extract text from the cropped license plate image.

```python
license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
```

- **`read_license_plate()`**:
  - Uses EasyOCR to extract text from the license plate image.
  - Calls `preprocess_license_plate()` to enhance the image for better OCR accuracy.
  - Calls `process_ocr_result()` to validate and format the detected text.

```python
def read_license_plate(license_plate_crop):
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        detections = reader.readtext(license_plate_crop)
        if detections:
            text, score = process_ocr_result(detections)
            if text:
                return text, score
        processed_plate = preprocess_license_plate(license_plate_crop)
        detections = reader.readtext(processed_plate)
        if detections:
            text, score = process_ocr_result(detections)
            if text:
                return text, score
        return None, None
    except Exception as e:
        print(f"Error in read_license_plate: {str(e)}")
        return None, None
```

- **`preprocess_license_plate()`**:
  - Enhances the license plate image using techniques like resizing, grayscale conversion, CLAHE, and adaptive thresholding.

```python
def preprocess_license_plate(license_plate_crop):
    height, width = license_plate_crop.shape[:2]
    license_plate_crop = cv2.resize(license_plate_crop, (width*2, height*2))
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh
```

- **`process_ocr_result()`**:
  - Processes the OCR results and checks if the detected text complies with the license plate format using `license_complies_format()`.
  - Formats the text using `format_license()`.

```python
def process_ocr_result(detections):
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None
```

- **`license_complies_format()`**:
  - Validates if the detected text follows the required license plate format.

```python
def license_complies_format(text):
    if len(text) != 7:
        return False
    # Check character format (e.g., AA11AAA)
    return True
```

- **`format_license()`**:
  - Formats the license plate text using character mapping dictionaries.

```python
def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_
```

##### **f. Store Results**
- The detected license plate text, bounding boxes, and confidence scores are stored in the `results` dictionary.

```python
results[frame_nmr][car_id] = {
    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
    'license_plate': {
        'bbox': [x1, y1, x2, y2],
        'text': license_plate_text,
        'bbox_score': score,
        'text_score': license_plate_text_score
    }
}
```

---

#### **Step 5: Write Results to CSV**
- After processing all frames, the `write_csv()` function is called to save the results to a CSV file.

```python
write_csv(results, './results.csv')
```

- **`write_csv()`**:
  - Writes the results (frame number, vehicle ID, bounding boxes, license plate text, and confidence scores) to a CSV file.

```python
def write_csv(results, output_path):
    try:
        with open(output_path, 'w') as f:
            f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')
            for frame_nmr in results.keys():
                for car_id in results[frame_nmr].keys():
                    if 'car' in results[frame_nmr][car_id].keys() and 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                        f.write(f"{frame_nmr},{car_id},[{results[frame_nmr][car_id]['car']['bbox'][0]} {results[frame_nmr][car_id]['car']['bbox'][1]} {results[frame_nmr][car_id]['car']['bbox'][2]} {results[frame_nmr][car_id]['car']['bbox'][3]}],[{results[frame_nmr][car_id]['license_plate']['bbox'][0]} {results[frame_nmr][car_id]['license_plate']['bbox'][1]} {results[frame_nmr][car_id]['license_plate']['bbox'][2]} {results[frame_nmr][car_id]['license_plate']['bbox'][3]}],{results[frame_nmr][car_id]['license_plate']['bbox_score']},{results[frame_nmr][car_id]['license_plate']['text']},{results[frame_nmr][car_id]['license_plate']['text_score']}\n")
        print(f"Results successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {str(e)}")
```

---

### **3. Program Ends**
- The program prints the total execution time and exits.

```python
print("Time taken: {:.2f} seconds".format(time.time() - start_time))
```

---

### **Summary of Execution Flow**
1. **Initialize Models**: Verify and load YOLOv8, license plate detector, and EasyOCR.
2. **Process Video Frames**:
   - Detect vehicles using YOLOv8.
   - Track vehicles using SORT.
   - Detect license plates using the license plate detector.
   - Associate license plates with vehicles.
   - Extract and validate license plate text using EasyOCR.
3. **Store Results**: Save the results in a CSV file.
4. **End Program**: Print the total execution time.

This detailed explanation should help you confidently explain the code during your review. Good luck!

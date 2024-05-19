##using tensorflow Lite
import cv2
import time
from tflite_runtime.interpreter import Interpreter
import numpy as np

FRAME_RATE = 1 / 5
def process_frame(frame):
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to the desired input shape of the model
    resized_frame = cv2.resize(frame_rgb, (224, 224))
    preprocessed_frame = resized_frame / 255.0

    return preprocessed_frame

interpreter = Interpreter(model_path='/home/pi/Downloads/Plant_Pest_Classifier.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference_on_frame(frame):
    processed_frame = process_frame(frame) 
    processed_frame = np.expand_dims(processed_frame, axis=0)     
    interpreter.set_tensor(input_details[0]['index'], processed_frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
    

cap = cv2.VideoCapture(0)
last_processed_time = time.time() 
class_labels = ['.ds_store','apple___apple_scab','apple___black_rot','apple___cedar_apple_rust','apple___healthy','blueberry___healthy','cherry_(including_sour)___healthy','cherry_(including_sour)___powdery_mildew',
                    'corn_(maize)___cercospora_leaf_spot gray_leaf_spot','corn_(maize)___common_rust_','corn_(maize)___healthy','corn_(maize)___northern_leaf_blight','grape___black_rot','grape___esca_(black_measles)',
                    'grape___healthy','grape___leaf_blight_(isariopsis_leaf_spot)','orange___haunglongbing_(citrus_greening)','peach___bacterial_spot','peach___healthy','pepper,_bell___bacterial_spot',
                    'pepper,_bell___healthy',  'potato___early_blight','potato___healthy','potato___late_blight','raspberry___healthy','soybean___healthy','squash___powdery_mildew','strawberry___healthy',
                    'strawberry___leaf_scorch','tomato___bacterial_spot','tomato___early_blight','tomato___healthy','tomato___late_blight','tomato___leaf_mold','tomato___septoria_leaf_spot','tomato___spider_mites two-spotted_spider_mite',
                    'tomato___target_spot','tomato___tomato_mosaic_virus','tomato___tomato_yellow_leaf_curl_virus'] 
last_processed_time = time.time() 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = time.time()
    if current_time - last_processed_time >= FRAME_RATE:
        last_processed_time = current_time
        output_data = run_inference_on_frame(frame)
        predicted_label = np.argmax(output_data, axis=1)[0]  
     
        predicted_class = class_labels[predicted_label]
        cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == 27:  
        break

cap.release()
cv2.destroyAllWindows()

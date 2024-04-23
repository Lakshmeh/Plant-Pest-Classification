
import streamlit as st
import cv2
import numpy as np




def process_frame(frame):
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    processed_frame = processed_frame / 255.0
    return processed_frame


def main():
    cap = cv2.VideoCapture(0)
    video_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            st.error("Error reading frame from webcam")
            break
        
        processed_frame = process_frame(frame)
        video_placeholder.image(processed_frame, channels="RGB", use_column_width=True, caption="Webcam Feed")
        
    cap.release()
    cap.release()

if __name__ == "__main__":
    main()
    
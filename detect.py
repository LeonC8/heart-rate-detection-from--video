import numpy as np
import cv2
import argparse

def face_detection_and_crop(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
 
    x, y, w, h = faces[0]
    return (x, y, w, h)  


def loadVideo(video_path):
    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()

        if ret is False:
            break

        image_sequence.append(frame[:, :, ::-1])

    video.release()

    return np.asarray(image_sequence), fps

def detect_face(video_path, output_size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
     
        face_region = face_detection_and_crop(frame)  

        if face_region is not None:
            x, y, w, h = face_region
            face_frame = frame[y:y+h, x:x+w]

            resized_frame = cv2.resize(face_frame, output_size)

            cut_frame = resized_frame[60:200, 60:200]
            frames.append(cut_frame)

    cap.release()
    return frames, fps

def main():
    parser = argparse.ArgumentParser(description='Process video for face detection.')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--detect_face', action='store_true', help='Enable face detection')

    args = parser.parse_args()

    if args.detect_face:
        images, fps = detect_face(video_path=args.video_path)
        images = np.array(images)
    else:
        images, fps = loadVideo(video_path=args.video_path)

    average_green_values = images[:, :, :, 1].mean(axis=(1, 2))

    freq_range = [0.7, 3]

    fft_values = np.fft.fft(average_green_values)
   
    fft_magnitudes = np.abs(fft_values)

    freqs = np.fft.fftfreq(len(average_green_values), d=1.0/fps)  

    low = (np.abs(freqs - freq_range[0])).argmin()
    high = (np.abs(freqs - freq_range[1])).argmin()

    relative_freqs = freqs[low:high]
    print("Broj otkucaja srca: " + str(int(relative_freqs[np.argmax(fft_magnitudes[low:high])]* 60)) + "/min")

if __name__ == "__main__":
    main()
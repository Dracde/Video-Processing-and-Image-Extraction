import cv2
import numpy as np
import os

# ============================================================
# Config
# ============================================================

# Folder paths
ASSETS_DIR = "assets"                                              #Base folder that holds all provided assets.
INPUT_VIDEOS_DIR = os.path.join(ASSETS_DIR, "Recorded Videos (4)") #Folder with the 4 raw videos.
INPUT_VIDEOS = ["alley.mp4", "office.mp4", "traffic.mp4", "singapore.mp4"]  #The 4 videos to process.

TALKING_VIDEO_PATH = os.path.join(ASSETS_DIR, "talking.mp4")       #Overlay video (Sub-task 3).
ENDSCREEN_VIDEO_PATH = os.path.join(ASSETS_DIR, "endscreen.mp4")   #End-screen video (Sub-task 5).
WATERMARK1_PATH = os.path.join(ASSETS_DIR, "watermark1.png")       #Watermark image 1 (Sub-task 4).
WATERMARK2_PATH = os.path.join(ASSETS_DIR, "watermark2.png")       #Watermark image 2 (Sub-task 4).
FACE_CASCADE_PATH = os.path.join(ASSETS_DIR, "face_detector.xml")  #Haar cascade model (Sub-task 2).

OUTPUT_DIR = "output_videos"                                       #Folder where processed videos are saved.

# Sub-task 1: Day / night detection & brightening
NIGHT_BRIGHTNESS_THRESHOLD = 100     #Average brightness below this = "nighttime".
BRIGHTNESS_BOOST = 60                #Value added to every pixel when brightening.
BRIGHTNESS_SAMPLE_COUNT = 15         #Number of frames sampled to decide day/night.

# Sub-task 2: Face blurring
FACE_SCALE_FACTOR = 1.3              #How much the image size is reduced at each scale.
FACE_MIN_NEIGHBORS = 5               #How many neighbours a detection needs to be kept.
FACE_BLUR_KERNEL = (51, 51)          #Gaussian blur kernel size used to blur each face (must be odd numbers).

# Sub-task 3: Talking-video overlay
OVERLAY_WIDTH_RATIO = 0.25           #Overlay width as a fraction of the main video's width.
OVERLAY_MARGIN = (20, 20)            #(x, y) position of the overlay's top-left corner.
OVERLAY_BORDER_COLOR = (0, 0, 0)     #Border colour drawn around the overlay (BGR).
OVERLAY_BORDER_THICKNESS = 3         #Border thickness in pixels.

# Sub-task 4: Watermarking
WATERMARK_SWITCH_SECONDS = 3         #How often (in seconds) the watermark alternates.


# ============================================================
# Sub-task 1: Detect if a video is daytime/nighttime, and brighten it if nighttime
# ============================================================


def compute_average_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #Convert frame to grayscale.
    return float(np.mean(gray))                      #Average pixel value = brightness (0-255).


def is_nighttime_video(video_path, sample_count=BRIGHTNESS_SAMPLE_COUNT):
    vid = cv2.VideoCapture(video_path)                              #Read video.
    total_no_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))        #Get total number of frames.
    total_no_frames = max(total_no_frames, 1)                       #Avoid divide-by-zero on empty videos.

    sample_indices = np.linspace(0, total_no_frames - 1, sample_count).astype(int)  #Pick evenly spread frames.

    brightness_values = []
    for frame_index in sample_indices:               #Loop through the sampled frame numbers.
        vid.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))  #Jump to that frame number.
        success, frame = vid.read()                   #Read that single frame.
        if success:
            brightness_values.append(compute_average_brightness(frame))  #Record its brightness.
    vid.release()                                     #Close the video file.

    avg_brightness = float(np.mean(brightness_values)) if brightness_values else 255.0  #Average of samples.
    return avg_brightness < NIGHT_BRIGHTNESS_THRESHOLD, avg_brightness  #True if darker than threshold.


def brighten_frame(frame, boost=BRIGHTNESS_BOOST):
    boost_matrix = np.full(frame.shape, boost, dtype=np.uint8)  #Matrix of the same size as the frame, filled with "boost".
    return cv2.add(frame, boost_matrix)                          #Add it to the frame (clips at 255, no wrap-around).


# ============================================================
# Sub-task 2: Detect faces in a frame and blur each one
# ============================================================


def load_face_detector(cascade_path=FACE_CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(cascade_path)   #Load pre-trained Haar cascade model.
    if face_cascade.empty():
        raise IOError(f"Could not load face cascade from: {cascade_path}")
    return face_cascade


def blur_faces(frame, face_cascade):
    faces = face_cascade.detectMultiScale(
        frame, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS)  #Perform face detection.

    for (x, y, w, h) in faces:                            #Loop through all the detected faces.
        face_region = frame[y:y + h, x:x + w]             #Crop out just the face.
        blurred_face = cv2.GaussianBlur(face_region, FACE_BLUR_KERNEL, 0)  #Blur that region.
        frame[y:y + h, x:x + w] = blurred_face            #Paste the blurred face back into the frame.

    return frame


# ============================================================
# Sub-task 3: Overlay talking.mp4 on the top-left corner
# ============================================================


def prepare_overlay_frame(talking_frame, main_frame_width):
    overlay_width = int(main_frame_width * OVERLAY_WIDTH_RATIO)   #25% of the main video's width.
    aspect_ratio = talking_frame.shape[0] / talking_frame.shape[1]        #Keep the same height/width ratio.
    overlay_height = int(overlay_width * aspect_ratio)

    resized = cv2.resize(talking_frame, (overlay_width, overlay_height))  #Shrink talking.mp4's frame down.
    cv2.rectangle(resized, (0, 0), (overlay_width - 1, overlay_height - 1),
                  OVERLAY_BORDER_COLOR, OVERLAY_BORDER_THICKNESS)  #Draw a border around it.
    return resized


def apply_overlay(main_frame, overlay_frame):
    x_margin, y_margin = OVERLAY_MARGIN               #Top-left position to paste the overlay.
    oh, ow = overlay_frame.shape[:2]

    x_end = min(x_margin + ow, main_frame.shape[1])           #Make sure the overlay fits inside the frame.
    y_end = min(y_margin + oh, main_frame.shape[0])

    main_frame[y_margin:y_end, x_margin:x_end] = overlay_frame[0:(y_end - y_margin), 0:(x_end - x_margin)]  #Paste it.
    return main_frame


# ============================================================
# Sub-task 4: Stamp a watermark on top of the frame, alternating every few seconds
# ============================================================


def load_watermark(watermark_path, frame_size):
    watermark = cv2.imread(watermark_path)      #Read the watermark image.
    if watermark is None:
        raise IOError(f"Could not load watermark image: {watermark_path}")
    return cv2.resize(watermark, frame_size)     #Resize it to match the video's frame size.


def choose_watermark(frame_count, fps, watermark1, watermark2):
    current_second = frame_count / fps                          #How many seconds into the video this frame is.
    block_number = int(current_second // WATERMARK_SWITCH_SECONDS)  #Which "few seconds" block we are in.
    return watermark1 if block_number % 2 == 0 else watermark2   #Even block = watermark1, odd block = watermark2.


def apply_watermark(frame, watermark):
    return cv2.add(frame, watermark)   #Add watermark onto frame (black areas add nothing, text areas brighten).


# ============================================================
# Main pipeline: runs Sub-tasks 1-5 on every input video, one video at a time
# ============================================================


def process_video(video_name, face_cascade):
    print(f"\nProcessing {video_name} ...")

    input_path = os.path.join(INPUT_VIDEOS_DIR, video_name)
    vid = cv2.VideoCapture(input_path)                          #Read video.
    if not vid.isOpened():
        print(f"  [ERROR] Could not open {input_path}")
        return

    fps = vid.get(cv2.CAP_PROP_FPS) or 30                       #Get the video's frame rate.
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))        #Get the video's width.
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))      #Get the video's height.
    frame_size = (frame_width, frame_height)                    #(width, height) used for resizing/output.

    night, brightness = is_nighttime_video(input_path)          #Sub-task 1: decide day or night.
    print(f"  Average brightness = {brightness:.2f} -> "
          f"{'NIGHTTIME (brightening)' if night else 'DAYTIME'}")

    talking_vid = cv2.VideoCapture(TALKING_VIDEO_PATH)   #Open the overlay video (Sub-task 3).

    watermark1 = load_watermark(WATERMARK1_PATH, frame_size)    #Sub-task 4: load both watermark images.
    watermark2 = load_watermark(WATERMARK2_PATH, frame_size)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0] + "_processed.avi")
    out = cv2.VideoWriter(output_path,               #Set the file name of the new video.
                           cv2.VideoWriter_fourcc(*'MJPG'),   #Set the codec.
                           fps,                       #Set the frame rate.
                           frame_size)                #Set the resolution (width, height).

    total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)   #Get total number of frames.
    for frame_count in range(0, int(total_no_frames)):    #To loop through all the frames.
        success, frame = vid.read()                       #Read a single frame from the video.
        if not success:
            break

        if night:
            frame = brighten_frame(frame)                 #Sub-task 1: brighten if nighttime.

        frame = blur_faces(frame, face_cascade)            #Sub-task 2: blur all detected faces.

        success_t, talking_frame = talking_vid.read()      #Sub-task 3: read a frame from talking.mp4.
        if success_t:
            overlay = prepare_overlay_frame(talking_frame, frame_width)
            frame = apply_overlay(frame, overlay)           #Paste the overlay onto the top-left corner.

        watermark = choose_watermark(frame_count, fps, watermark1, watermark2)  #Sub-task 4: pick watermark for this second.
        frame = apply_watermark(frame, watermark)           #Stamp the chosen watermark on top.

        out.write(frame)                                    #Save processed frame into the new video.

    vid.release()                                           #Close the input video.
    talking_vid.release()                                   #Close the overlay video.

    endscreen_vid = cv2.VideoCapture(ENDSCREEN_VIDEO_PATH)   #Sub-task 5: open the end-screen video.
    endscreen_total = endscreen_vid.get(cv2.CAP_PROP_FRAME_COUNT)   #Get total number of end-screen frames.
    for frame_count in range(0, int(endscreen_total)):              #Loop through all end-screen frames.
        success, frame = endscreen_vid.read()                       #Read a single end-screen frame.
        if not success:
            break
        frame = cv2.resize(frame, frame_size)                       #Match the main video's resolution.
        out.write(frame)                                            #Append it after the last processed frame.
    endscreen_vid.release()                                         #Close the end-screen video.

    out.release()                                                   #Finish writing the output video.
    print(f"  Saved to: {output_path}")


def main():
    face_cascade = load_face_detector()          #Load pre-trained Haar cascade model (once, reused for all videos).
    for video_name in INPUT_VIDEOS:               #Loop through all 4 input videos.
        process_video(video_name, face_cascade)
    print("\nAll videos processed.")


if __name__ == "__main__":
    main()

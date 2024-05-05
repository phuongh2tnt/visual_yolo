import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/content/drive/My Drive/AI/el/best.pt')

# Open the video file
video_path = '/content/drive/My Drive/AI/el/videotestcoMinh/coMinh7.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize frame count
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Save the frame
        #cv2.imwrite("frame_%d.jpg" % frame_count, annotated_frame)
       # print("Frame %d saved successfully." % frame_count)
        #-----Luu
        save_dir = '/content/drive/My Drive/AI/el/coMinh'
        cv2.imwrite(os.path.join(save_dir, "frame_%d.jpg" % frame_count), annotated_frame)
        #print("Frame %d saved successfully." % frame_count)
        # Increment frame count
        frame_count += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

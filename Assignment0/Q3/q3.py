import cv2
import matplotlib.pyplot as plt
import numpy as np

# lower_blue = np.array([10, 110, 10])     ##[R value, G value, B value]
# upper_blue = np.array([90, 190, 90])

lower_blue = np.array([210, 210, 210])     ##[R value, G value, B value]
upper_blue = np.array([255, 255, 255])

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_mask(img, low_thresh, high_thresh):
    mask = cv2.inRange(img, low_thresh, high_thresh)
    masked_img = np.copy(img)
    masked_img[mask != 0] = [0, 0, 0]
    return masked_img, mask

def mask_background(bg_img, mask):
    bg_img[mask == 0] = [0, 0, 0]
    return bg_img

def merge_bg_fg(fg, bg):
    final_img = fg + bg
    return final_img

def apply_chroma_key(fg_path, bg_path):
    cap_fg = cv2.VideoCapture(fg_path)
    cap_bg = cv2.VideoCapture(bg_path)

    if cap_fg.isOpened() == False:
        print("Error opening video file")

    if cap_bg.isOpened() == False:
        print("Error opening video file")

    frames = []

    while cap_fg.isOpened() and cap_bg.isOpened():
        ret_fg, fg = cap_fg.read()
        ret_bg, bg = cap_bg.read()

        if ret_fg == True and ret_bg == True:

            masked_img, mask = extract_mask(fg, lower_blue, upper_blue)
            masked_bg = mask_background(bg, mask)
            final_img = merge_bg_fg(masked_img, masked_bg)
            frames.append(final_img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break
    
    cap_fg.release()
    cap_bg.release()

    return frames

def convert_frames_to_video(frames, output_file, fps): 
    size =  frames[0].shape
    size = (size[1], size[0])
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def main():
    frames = apply_chroma_key('shame_fg.mp4', 'shame_bg.mp4')
    convert_frames_to_video(frames, 'final_shame.mp4', 24)

main()

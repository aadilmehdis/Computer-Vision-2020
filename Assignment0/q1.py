import cv2
from glob import glob


parser = argparse.ArgumentParser(description='Converting Frames to Video and Vice Versa')
parser.add_argument('--in', dest='input', required=True, help = "[--in \path\to\input\directory]")
parser.add_argument('--out', dest='out', required=True, help="[--out \parh\to\output\directory]")

def frames_to_video(input_path, output_path, fps):
    '''
        Function to Concatenate given frames and fps into a video file.

        Input Arguments
        input_path  : Path to the input directory containing input frames
        output_path : Path to the output directory containing the video file
        fps         : Frames per Second of the output video

        Return
        Boolean     : True is Video written successfully, False if writing is not successful.
    '''

    if not os.path.isdir(input_path):
        raise OSError(2, 'No such file or directory', input_path)
        return False

    if not os.path.isdir(output_path):
        os.makedirs(output_path)


    image_files = sorted(glob.glob(input_path))

    frames = []

    for f in image_files:
        frame = cv2.imread(f)
        height, width, channels = frame.shape
        size = (width, height)
        frames.append(frame)

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()

    return True

     

def video_to_frames():
    '''
        Function to extract out frames from a given video and fps.

        Input Arguments

        Return
    '''
    pass
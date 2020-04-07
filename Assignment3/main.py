from gui import GUI
import sys
import os
import cv2 as cv

if __name__ == '__main__':    
    image_name = sys.argv[1].split('/')[-1].split('.')[0]
    output_dir = 'outputs/'+image_name
    os.mkdir(output_dir)
    img = cv.imread(sys.argv[1])#cv.cvtColor(cv.imread(sys.argv[1]), cv.COLOR_BGR2RGB)
    gui = GUI(img)
    gui.run()
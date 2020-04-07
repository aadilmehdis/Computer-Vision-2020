# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys
import os


from config import *
from grbcut import GrabCut
# from tre import GrabCut

iterator = 1

class GUI:
    def __init__(self, input_image, brush_thickness=3):
        self.img = input_image
        self.img2 = input_image.copy()
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8) 
        self.output = np.zeros(self.img.shape,np.uint8)
        self.rect = (0,0,1,1)
        self.drawing = False         # flag for drawing curves
        self.rectangle = False       # flag for drawing rect
        self.rect_over = False       # flag to check if rect drawn
        self.rect_or_mask = 100      # flag for selecting rect or mask mode
        self.value = DRAW_FG         # drawing initialized to FG
        self.thickness = 3           # brush thickness
        self.iterator = 1

    def onmouse(self, event, x, y, flags, param):
        global ix, iy
        # Draw Rectangle
        if event == cv.EVENT_LBUTTONDOWN:
            self.rectangle = True
            ix, iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img, (ix,iy), (x,y), BLUE, 2)
                self.rect = (min(ix,x), min(iy,y), abs(ix-x), abs(iy-y))
                self.rect_or_mask = 0

        elif event == cv.EVENT_LBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (ix,iy), (x,y), BLUE, 2)
            self.rect = (min(ix,x), min(iy,y), abs(ix-x), abs(iy-y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # draw touchup curves

        if event == cv.EVENT_MBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1) # flag for drawing curves

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask,( x,y), self.thickness, self.value['val'], -1)

        elif event == cv.EVENT_MBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)


    def reset_params(self):
        print("resetting \n")
        self.rect = (0,0,1,1)
        self.drawing = False
        self.rectangle = False
        self.rect_or_mask = 100
        self.rect_over = False
        self.value = DRAW_FG
        self.img = self.img2.copy()
        self.mask = np.zeros(self.img.shape[:2],dtype = np.uint8)
        self.output = np.zeros(self.img.shape,np.uint8)     

    def run(self):
        cv.namedWindow('output')
        cv.namedWindow('input')
        cv.setMouseCallback('input',self.onmouse)
        cv.moveWindow('input',self.img.shape[1]+10,90)

        while(1):

            cv.imshow('output',self.output)
            cv.imshow('input',self.img)
            k = cv.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            elif k == ord('0'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = DRAW_BG
            elif k == ord('1'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = DRAW_FG
            elif k == ord('2'): # PR_BG drawing
                self.value = DRAW_PR_BG
            elif k == ord('3'): # PR_FG drawing
                self.value = DRAW_PR_FG
            elif k == ord('s'): # save image
                bar = np.zeros((self.img.shape[0],5,3),np.uint8)
                res = np.hstack((self.img2,bar,self.img,bar,self.output))
                cv.imwrite(output_dir+'/'+ image_name+'_{}.png'.format(self.iterator),res)
                self.iterator += 1
                print(" Result saved as image \n")
            elif k == ord('r'): # reset everything
                self.reset_params()
            elif k == ord('n'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                print(self.rect)
                if (self.rect_or_mask == 0):         # grabcut with rect
                    # bgdmodel = np.zeros((1,65),np.float64)
                    # fgdmodel = np.zeros((1,65),np.float64)
                    # cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT)
                    grab_cut = GrabCut(self.img2, self.mask, self.rect, k_gmm_components=5)
                    grab_cut.run()
                    self.rect_or_mask = 1
                elif self.rect_or_mask == 1:         # grabcut with mask
                    # bgdmodel = np.zeros((1,65),np.float64)
                    # fgdmodel = np.zeros((1,65),np.float64)
                    # cv.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK)
                    grab_cut.run()

            mask2 = np.where((self.mask==1) + (self.mask==3),255,0).astype('uint8')
            self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)

        cv.destroyAllWindows()


if __name__ == '__main__':
    # img = cv.imread('messi5.jpg')
    
    image_name = sys.argv[1].split('/')[-1].split('.')[0]
    output_dir = 'outputs/'+image_name
    os.mkdir(output_dir)
    img = cv.imread(sys.argv[1])#cv.cvtColor(cv.imread(sys.argv[1]), cv.COLOR_BGR2RGB)
    gui = GUI(img)
    gui.run()

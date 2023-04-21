from cgitb import reset
import cv2
import harris
import shitomasi
import sift
import fast
import orb
import brute_force
import flann

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    while cap.isOpened(): 
        ret, frame = cap.read() #returns frame and also ret to confirm if it reading worked
        ret1, frame1 = cap1.read()
        rbg_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        #DETECTOR
        #harris.livecapture(frame1)
        #shitomasi.livecapture(frame1)
        #sift.livecapture(frame1)
        #fast.livecapture(frame1)
        #orb.livecapture(frame1)


        #MATCHER
        #brute_force.livecapture(frame,frame1)
        #flann.livecapture(frame,frame1)
        
        
            
        

        if cv2.waitKey(1) == ord('q'):
            break
 
       
         

        
        
    cap.release()
    cv2.destroyAllWindows()

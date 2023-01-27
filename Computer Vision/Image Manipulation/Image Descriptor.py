import cv2
#Look for the following files for reference:
# Dectors and/or Descriptors
#1. Harris Corner Detector - harris.py
import harris
#harris.main('../deliverables/images/face_orig.jpg')

#2. Shit Tomasi Corner Detector - shitomasi.py
import shitomasi
#shitomasi.main('../deliverables/images/cat_orig.jpg')

#3. SIFT - sift.py
import sift
#sift.main('../deliverables/images/building_orig.jpg')

#4. FAST - fast.py
import fast
#fast.main('../deliverables/images/building_orig.jpg')

#5. ORB - orb.py
import orb
#orb.main('../deliverables/images/building_orig.jpg')

# Matching
#6. Brute Force Matcher - brute_force.py
import brute_force
img = brute_force.bfm('../deliverables/images/building_orig.jpg','../deliverables/images/building_tilt.jpg')
cv2.imshow('Brute Force', img)


#7. FLANN based Matcher - flann.py
import flann

#img = flann.flann('../deliverables/images/cat_orig.jpg','../deliverables/images/cat_tilt.jpg')
#cv2.imshow('FLANN', img)
#cv2.waitKey()


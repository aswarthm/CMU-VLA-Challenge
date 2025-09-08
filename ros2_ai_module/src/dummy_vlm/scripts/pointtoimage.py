import cv2

# Load your image
img = cv2.imread('/home/aswarth/CMU-VLA-Challenge/images/3_image.png')

# Coordinates of the point
point_x = 265*1080//1000
point_y = 476*640//1000

# Draw a red circle at the point (BGR color: (0,0,255))
cv2.circle(img, (point_x, point_y), radius=8, color=(0, 0, 255), thickness=-1)
print(point_x, point_y)

# Show the image
cv2.imshow('Image with Point', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
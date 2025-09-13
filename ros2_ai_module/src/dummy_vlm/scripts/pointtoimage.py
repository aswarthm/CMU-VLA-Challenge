import cv2

# Load your image
img = cv2.imread('/home/aswarth/CMU-VLA-Challenge/images/1_image.png')

points = [
            [485, 195], [420, 350], [550, 450], [550, 600], [485, 200]
            ]

def plot(x, y, idx):
    point_x = x*1920//1000
    point_y = y*640//1000
    cv2.putText(img, str(idx), (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    print(point_x, point_y)

for idx, point in enumerate(points):
    plot(point[1], point[0], idx)
# Show the image
cv2.imshow('Image with Point', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
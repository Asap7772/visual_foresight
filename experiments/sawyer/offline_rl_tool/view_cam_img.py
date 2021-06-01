import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

im2 = cv2.imread("im_0.png")
im2 = cv2.resize(im2, (512, 512), interpolation=cv2.INTER_CUBIC)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    frame= cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("test", cv2.addWeighted((frame[::-1, ::-1]), 0.4, im2,0.6, 0))

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

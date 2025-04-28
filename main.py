
import numpy as np
import cv2 as cv

img = cv.imread('Images/lane2.jpg')
vid = cv.VideoCapture('Videos/straight_lane.mp4')

# Define trapezoidal ROI for lane focus
height, width = img.shape[:2]
roi = np.array([[
    (width * 0.1, height),  # Bottom left
    (width * 0.45, height * 0.6),  # Top left
    (width * 0.55, height * 0.6),  # Top right
    (width * 0.9, height)  # Bottom right
]], dtype=np.int32)

#Preprocesses image for lane detection by converting to grayscale, applying Gaussian blur, and Canny edge detection
def processImage(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 100, 200)
    return canny

#Applies mask on the (processed) image to isolate ROI.
def region_of_interest(img, roi):
    mask = np.zeros_like(img)
    color = 255
    cv.fillPoly(mask, roi, color)
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

#Draws lines on a blank image
def draw_lines(img, lines, color = [255,0,0], thickness = 3):
    line_image = np.zeros_like(img)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image, (x1,y1), (x2,y2), color=[255,0,0], thickness=3)
    return line_image

#Processes detected Hough lines and separates them into left/right lanes
def slope(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0,1,3,2]

    left_lines = []
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1==x2:
                pass
            else:
                m = (y2-y1) / (x2-x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m,c))
                else:
                    right_lines.append((m,c))

    if left_lines and right_lines:
        left_avg = np.average(left_lines, axis=0)
        right_avg = np.average(right_lines, axis =0)
        for slope, intercept in [left_avg, right_avg]:
            rows, cols = img.shape[:2]
            y1 = int(rows)
            y2 = int(rows*0.6)
            x1 = int((y1-intercept)/slope)
            x2= int((y2-intercept)/slope)
            poly_vertices.append((x1,y1))
            poly_vertices.append((x2,y2))
            draw_lines(img, np.array([[[x1,y1,x2,y2]]]))

        poly_vertices = [poly_vertices[i] for i in order]
        cv.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0,255,0))
        return cv.addWeighted(image, 0.7, img, 0.4, 0)

#Detects line segments in an edge image using Hough Transform, then processes them into coherent lane markings.
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]),
                           minLineLength=min_line_len, maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = slope(line_img, lines)
    return line_img

def weighted_img(img, initial_img, alpha=0.1, beta=1, gamma=0.):
    return cv.addWeighted(initial_img, alpha, img, beta, gamma)


#Executes the complete lane detection pipeline on a single image.
def readImage(img):
    canny = processImage(img)
    masked_image = region_of_interest(canny, np.array([roi]))
    hough_lines_img = hough_lines(masked_image, 1, np.pi/180, 20, 20, 180)
    output = weighted_img(hough_lines_img, img, 0.8, 1, 0.)

    cv.imshow('Original Image', img)
    cv.imshow('Output', output)

#Executes the complete lane detection pipeline on a video.
def readVideo(video):
    while vid.isOpened():
        ret, frame = vid.read()

        if not ret:
            break

        canny = processImage(frame)
        masked_image = region_of_interest(canny, np.array([roi]))
        hough_lines_img = hough_lines(masked_image, 1, np.pi / 180, 20, 20, 180)
        output = weighted_img(hough_lines_img, frame, 0.8, 1, 0.)
        cv.imshow('Video', output)
        if cv.waitKey(25) & 0xFF == ord('q'):  # Press Q to exit
            break


readImage(img)
#readVideo(vid)
#vid.release()
#Press D to exit
cv.waitKey(0)
cv.destroyAllWindows()
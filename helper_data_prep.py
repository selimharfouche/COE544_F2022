# Helpers
# function that merges 2 rectangles
# takes as input two bounding rectangles and returns them merged
import cv2
def union(a, b):
 x = min(a[0], b[0])
 y = min(a[1], b[1])
 w = max(a[0] + a[2], b[0] + b[2]) - x
 h = max(a[1] + a[3], b[1] + b[3]) - y
 return (x, y, w, h)


def convert_BW(image):
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV)
    return thresh
###########################################################################
###########################################################################
# Feature #1: Aspect Ratio

# images are stored in 2D arrays
# image.shape is (n,m). So image.shape[0] is n.

# aspect ratio (in decimal format)
def aspect_ratio(cropped_image):
    # number of columns / number of rows
    ar = ((float)(cropped_image.shape[1] / cropped_image.shape[0]))
    return ar

# aspect ratio (in fraction format)
def calculate_aspect(width: int, height: int) -> str:
    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)

    r = gcd(width, height)
    x = int(width / r)
    y = int(height / r)

    print(f"{x}:{y}")
    print()
    return
###########################################################################
###########################################################################

# Feature 2: Pixel Count, Pixel Percent
# takes as input the cropped image
# returns the percentage of black pixels in it

###########################################################################
def top_half_img(cropped_image):
    top_half = cropped_image[int(0):int(cropped_image.shape[0] / 2), int(0):int(cropped_image.shape[1])]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(top_half)

    # get pixel count of image
    height, width = top_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = int((cntBlack / cntPixels) * 100)

    return percent_black

###########################################################################
def lower_half_img(cropped_image):
    lower_half = cropped_image[int(cropped_image.shape[0] / 2):int(cropped_image.shape[0]), int(0):int(cropped_image.shape[1])]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(lower_half)

    # get pixel count of image
    height, width = lower_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    return percent_black

###########################################################################
def right_half_img(cropped_image):
    right_half = cropped_image[int(0):int(cropped_image.shape[0]), int(cropped_image.shape[1] / 2):int(cropped_image.shape[1])]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(right_half)

    # get pixel count of image
    height, width = right_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    return percent_black

###########################################################################
def left_half_img(cropped_image):
    left_half = cropped_image[int(0):int(cropped_image.shape[0]), int(0):int(cropped_image.shape[1] / 2)]

    # get all non black Pixels
    cntNotBlack = cv2.countNonZero(left_half)

    # get pixel count of image
    height, width = left_half.shape
    cntPixels = height * width

    # compute all black pixels
    cntBlack = cntPixels - cntNotBlack
    percent_black = (cntBlack / cntPixels) * 100

    return percent_black


###########################################################################
###########################################################################
# Feature 3: projection histogram
def histogram(cropped_image):
    cropped_image[cropped_image == 0] = 1
    cropped_image[cropped_image == 255] = 0

    # Calculate horizontal projection
    hor_proj = np.sum(cropped_image, axis=1)

    height, width = cropped_image.shape

    blankImage = np.zeros((height, width), np.uint8)

    # Draw a line for each row
    for row in range(height):
        cv2.line(blankImage, (0, row), (int(hor_proj[row] * width / height), row), (255, 255, 255), 1)

    # Save result
    blankImage = cv2.resize(blankImage, (128, 128), interpolation=cv2.INTER_AREA)
    imgName = "img" + str(counter) + ".png"
    cv2.imwrite(os.path.join("test5", imgName), blankImage)

    return blankImage

###########################################################################
# Feature 5
# Pixel intensity
def pixel_intensity(cropped_image):
    n_samples = len(cropped_image)
    cropped_image_reshaped = cropped_image.reshape((n_samples, -1))
    return cropped_image_reshaped
###########################################################################
###########################################################################
###########################################################################
###########################################################################

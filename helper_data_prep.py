
image_path = "Images/0/AAAAA.png"


from PIL import Image


def black_and_white(image):
    col =image
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    return bw



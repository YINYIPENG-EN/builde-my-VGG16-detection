from PIL import Image
from vgg16_detection import VGG16

vgg16 = VGG16()
while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = vgg16.detect_image(image)
        r_image.show()
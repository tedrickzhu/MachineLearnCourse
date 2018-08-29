import cv2
import numpy as np
import Image
import cv2.cv as cv


def image_joint(image_list, opt):  # opt= [1,4]一行四列
    image_num = len(image_list)
    image_size = image_list[0].size
    height = image_size[1]
    width = image_size[0]

    new_height = opt[0] * height
    new_width = opt[1] * width

    new_img = Image.new('RGB', (new_width, new_height), 255)

    x = y = 0
    x_count = 0
    y_count = 0
    for img in image_list:

        print 'ccc'
        if y_count < opt[0]:
            if x_count < opt[1]:
                new_img.paste(img, (x, y))
                x += width
                x_count += 1
                print 'hi'
                continue
            print 'hellp'
            x = 0
            x_count = 0
            y += height
            new_img.paste(img, (x, y))
            x += width
    new_img.show()

    return new_img
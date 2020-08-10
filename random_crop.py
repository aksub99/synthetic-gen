import os
import json
import bisect
import skimage
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont


alphabets = {
    'a': ['A', '(A)', 'a', '(a)'],
    'b': ['B', '(B)', 'b', '(b)'],
    'c': ['C', '(C)', 'c', '(c)'],
    'd': ['D', '(D)', 'd', '(d)'],
    'e': ['E', '(E)', 'e', '(e)'],
    'f': ['F', '(F)', 'f', '(f)'],
    'g': ['G', '(G)', 'g', '(g)'],
    'h': ['H', '(H)', 'h', '(h)'],
    # 'i': ['I', '(I)', 'i', '(i)'],
    # 'j': ['J', '(J)', 'j', '(j)'],
    # 'k': ['K', '(K)', 'k', '(k)']
}

alpha_to_id = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    # 'i': 8,
    # 'j': 9,
    # 'k': 10,
}

annotations = {
    'images': [],
    'type': 'instances',
    'annotations': [],
    'categories': [{"supercategory": "none", "id": 0, "name": "a"}, {"supercategory": "none", "id": 1, "name": "b"}, {"supercategory": "none", "id": 2, "name": "c"}, {"supercategory": "none", "id": 3, "name": "d"}, {"supercategory": "none", "id": 4, "name": "e"}, {"supercategory": "none", "id": 5, "name": "f"}, {"supercategory": "none", "id": 6, "name": "g"}, {"supercategory": "none", "id": 7, "name": "h"}]
}

ctr = 0

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

def skimage_noisy(image):
    img = image/255.0
    gimg = skimage.util.random_noise(img, mode="gaussian")
    gimg = (gimg * 255).astype(np.uint8)
    return gimg

def noisy(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image / 255.0 + gauss
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def random_paste_text(category, base_img, text):
    global ctr
    margin_x = np.random.randint(0, 6)
    margin_y = np.random.randint(0, 6)

    font_type = random.choice(os.listdir("fonts"))
    font_size = np.random.randint(11, 25)

    font = ImageFont.truetype(os.path.join("fonts", font_type), font_size)
    is_noisy = np.random.choice([True, False], p=[0.3, 0.7])
    if is_noisy:
        base = Image.fromarray(skimage_noisy(base_img))
    else:
        base = Image.fromarray(base_img)

    draw = ImageDraw.Draw(base)

    text_size = draw.textsize(text, font)

    x_min = margin_x + random.randint(0, 5)
    y_min = margin_y + random.randint(0, 5)

    x_max = base_img.shape[1] - text_size[0] - margin_x + random.randint(-4, 1)
    y_max = base_img.shape[0] - text_size[1] - margin_y + random.randint(-4, 1)

    x_med = random.randint(x_min, x_max)
    y_med = random.randint(y_min, y_max)

    x = np.random.choice([x_min, x_med, x_max], p=[0.4, 0.2, 0.4]).item()
    y = np.random.choice([y_min, y_med, y_max], p=[0.4, 0.2, 0.4]).item()

    crop = base_img[y - margin_y: y + text_size[1] + margin_y, x - margin_x: x  + text_size[0] + margin_x]
    if crop.mean() <= 127:
        # Light colors
        code = (random.randint(200, 256), random.randint(200, 256), random.randint(200, 256))
    else:
        # Dark colors
        code = (random.randint(0, 51), random.randint(0, 51), random.randint(0, 51))

    draw.text((x, y), text, code, font)

    base.save("images/{}.jpg".format(ctr))
    annotations['images'].append({"file_name": "{}.jpg".format(ctr), "height": base.size[1], "width": base.size[0], "id": ctr})
    annotations['annotations'].append({"area": (text_size[0] + 2 * margin_x) * (text_size[1] + 2 * margin_y), "iscrowd": 0, "image_id": ctr, "bbox": [x - margin_x, y - margin_y, text_size[0] + 2 * margin_x, text_size[1] + 2 * margin_y], "category_id": alpha_to_id[category], "id": ctr, "ignore": 0, "segmentation": []})
    ctr += 1


def random_paste_image(category, base_img, paste_img, crop_height, crop_width):
    global ctr

    max_x = base_img.shape[1] - crop_width
    max_y = base_img.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    base = Image.fromarray(base_img)
    paste = Image.fromarray(paste_img)

    base.paste(paste, (x, y))

    base.save("images/{}.jpg".format(ctr))
    annotations['images'].append({"file_name": "{}.jpg".format(ctr), "height": base.size[1], "width": base.size[0], "id": ctr})
    annotations['annotations'].append({"area": paste.size[0] * paste.size[1], "iscrowd": 0, "image_id": ctr, "bbox": [x, y, paste.size[0], paste.size[1]], "category_id": alpha_to_id[category], "id": ctr, "ignore": 0, "segmentation": []})
    ctr += 1

path="../full_dataset/images_train"

sizes = [64, 128, 256, 512]
files = os.listdir(path)

for i in range(1000):
    f = random.choice(files)
    example_image = np.array(Image.open(os.path.join(path, f)))
    min_side = min(example_image.shape[0], example_image.shape[1])

    index = bisect.bisect_left(sizes, min_side)
    if index == 0:
        continue
    else:
        crop_size = sizes[index - 1]

    random_crop = get_random_crop(example_image, crop_size, crop_size)

    for alphabet in alphabets:
        category = alphabet
        for tp in ['alphabets', 'alphabets_inv']:
            paste_img = np.array(Image.open("{}/{}.png".format(tp, random.choice(alphabets[alphabet]))))
            paste_text = random.choice(alphabets[alphabet])

            # random_paste_image(category, random_crop, paste_img, paste_img.shape[0], paste_img.shape[1])
            random_paste_text(category, random_crop, paste_text)

with open('anns.json', 'w') as outfile:
    json.dump(annotations, outfile)

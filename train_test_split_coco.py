import os
import json


anns =  json.load(open("anns.json"))
anns_train = {"images": [], "type": "instances", "annotations": [], "categories": None}
anns_test = {"images": [], "type": "instances", "annotations": [], "categories": None}

anns_train["categories"] = anns["categories"]
anns_test["categories"] = anns["categories"]

for f in os.listdir("images_train"):
    is_query_image = False
    for image in anns["images"]:
        if image["file_name"] == f:
            anns_train["images"].append(image)
            image_id = image["id"]
            is_query_image = True
    if is_query_image:
        for annotation in anns["annotations"]:
            if annotation["image_id"] == image_id:
                anns_train["annotations"].append(annotation)

with open('output_train.json', 'w') as outfile:
    json.dump(anns_train, outfile)

for f in os.listdir("images_test"):
    is_query_image = False
    for image in anns["images"]:
        if image["file_name"] == f:
            anns_test["images"].append(image)
            image_id = image["id"]
            is_query_image = True
    if is_query_image:
        for annotation in anns["annotations"]:
            if annotation["image_id"] == image_id:
                anns_test["annotations"].append(annotation)

with open('output_test.json', 'w') as outfile:
    json.dump(anns_test, outfile)

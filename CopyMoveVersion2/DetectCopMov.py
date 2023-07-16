import cv2
import matplotlib.pyplot as plt
import numpy as np

config_file=r"CopyMoveVersion2\ConfigFiles\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"     # common object in context
frozen_model=r"CopyMoveVersion2\ConfigFiles\frozen_inference_graph.pb"
labels_file=r"CopyMoveVersion2\ConfigFiles\labels.txt"


classLables=[]
with open(labels_file,'rt') as fpt:
    classLables=fpt.read().rstrip("\n").split("\n")



model = cv2.dnn_DetectionModel(frozen_model,config_file)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

img=cv2.imread(r"images\an.jpeg")

# plt.figure("Raw Image")
# plt.imshow(img)

ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
print(confidence)
print(bbox)

Object_Boxes=[]
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    Object_Boxes.append(boxes)


plt.figure("Boxes Image")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.savefig(r'CopyMoveVersion2\ObjDetected\Boxes.png')


for i in range(len(Object_Boxes)):
    for j in range(i+1,len(Object_Boxes)):
        box1=Object_Boxes[i]
        box2=Object_Boxes[j]

        image1 = img[box1[1]:box1[1]+box1[3], box1[0]:box1[0]+box1[2]]
        image2 = img[box2[1]:box2[1]+box2[3], box2[0]:box2[0]+box2[2]]

        resized_image1 = cv2.resize(image1, (100, 100))
        resized_image2 = cv2.resize(image2, (100, 100))

        gray1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

        # Calculate the histograms of the grayscale images
        hist1, _ = np.histogram(gray1.flatten(), bins=256, range=[0, 256])
        hist2, _ = np.histogram(gray2.flatten(), bins=256, range=[0, 256])

        # Normalize the histograms
        hist1 = hist1.astype(float) / np.sum(hist1)
        hist2 = hist2.astype(float) / np.sum(hist2)

        # Calculate the Histogram Intersection
        hist_intersection = np.minimum(hist1, hist2)
        similarity_score = np.sum(hist_intersection)

        if similarity_score >=0.70:
            cv2.rectangle(img,box1,(0,255,0),2)
            cv2.rectangle(img,box2,(0,255,0),2)

            # Print the similarity score
            print(f"Histogram Intersection Score: {similarity_score}")
            plt.figure("Similar Image")
            plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            plt.savefig(r'CopyMoveVersion2\ObjDetected\Boxes.png')
print(similarity_score)

        



plt.show()
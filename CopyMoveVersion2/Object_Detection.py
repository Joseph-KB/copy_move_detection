import cv2  # pip install opencv-python
import matplotlib.pyplot as plt     # pip install matplotlib


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

img=cv2.imread(r"images\prjct.jpg")

# plt.figure("Raw Image")
# plt.imshow(img)

ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
print(confidence)
print(bbox)

font_scale=3
font = cv2.FONT_HERSHEY_PLAIN

for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
#   cv2.putText(img,classLables[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)

plt.figure("Boxes Image")
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.savefig(r'CopyMoveVersion2\ObjDetected\Boxes.png')




plt.show()
import cv2, requests, io, json, pyrebase, os
from difflib import SequenceMatcher
from datetime import date

firebaseConfig = {"firebase Config"}

firebase=pyrebase.initialize_app(firebaseConfig)
db=firebase.database()
users=db.child("data").get()
values = users.val()
fine_list = []
fine_list_date = []
today = date.today()
d1 = today.strftime("%d")
key = list(values.keys())
thres = 0.3
cap = cv2.VideoCapture("Video1.mp4")   #input type
cap.set(3, 1280)
cap.set(4, 720)
lis=['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','street sign','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','hat','backpack','umbrella','shoe','eye glasses','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','plate','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','mirror','dining table','window','desk','toilet','door','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','toothbrush''hair brush']
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weigthsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.2, 127.5, 127.5))
net.setInputSwapRB(True)
i=0
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    for y in range(len(classIds)):
        ciid = lis[(classIds[y-1].item())-1].upper()
        print(ciid)
        clis= ["CAR", "BUS", "MOTORCYCLE", "BICYCLE", "TRUCK"]
        if ciid in clis:
            v = 'Frame'+str(i)+'.jpg'
            i += 1
            cv2.imwrite(v,img)
            img = cv2.imread(v)
            height, width, _ = img.shape
            roi = img
            url_api = "https://api.ocr.space/parse/image"
            _, compressedimage = cv2.imencode(".jpg", roi, [1, 90])
            file_bytes = io.BytesIO(compressedimage)
            result = requests.post(url_api,
                          files = {"screenshot.jpg": file_bytes},
                          data = {"apikey": "api key",
                                  "language": "eng"})
            result = result.content.decode()
            result = json.loads(result)
            parsed_results = result.get("ParsedResults")[0]
            text_detected = parsed_results.get("ParsedText")
            print(text_detected)
            
            for num in key:
                s = SequenceMatcher(None, num, text_detected)
                match = int(s.ratio()*100)
                if match > 50:
                    number = num
                    today = date.today()
                    d2 = today.strftime("%d")
                    if d1 != d2:
                        fine_list.clear()
                        fine_list_date.clear()
                    if number not in fine_list:
                        fine_list.append(number)
                        fine_list_date.append(d2)
                        C_fine=db.child("data").child(number).child("Fine").get()
                        amount = C_fine.val()+500
                        db.child("data").child(number).update({"Fine":amount})
                        T_mail=values[num]["email"]
                        T_name=values[num]["Vehicle Owner"]
                        T_type = values[num]["type"]
                        T_fine=str(amount)
                        message="Your parked your "+T_type+" with number "+number+" in no parking area. Fine amount 500 has imposed on your vehicle. Your total fine amount is "+T_fine
                        print(message)
                        report = {}
                        report["value1"] = T_mail
                        report["value2"] = number
                        report["value3"] = T_fine
                        requests.post("trigger link", data=report)    
                        report.clear()
            os.remove(v)
            
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
            cv2.putText(img,classNames[classId-1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence*100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Output", img)
    cv2.waitKey(1)

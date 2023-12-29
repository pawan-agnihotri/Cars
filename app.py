from flask import Flask, render_template, request
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import Tracker
tracker = Tracker()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text input value from the form
        text_input = request.form['text_input']
        model = YOLO('yolov8m.pt')

        # Process the input (you can perform any logic here)
        processed_text = f"You entered: {text_input}"
        #stream = CamGear(source='https://www.youtube.com/watch?v=hIglzOpSK3E', stream_mode=True, logging=True).start()
        stream = CamGear(source=text_input, stream_mode=True, logging=True).start()
        my_file = open("coco.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")
        # print(class_list)
        count = 0
        tracker = Tracker()

        while True:

            frame = stream.read()
            count += 1
            if count % 2 != 0:
                continue

            try:
                frame = cv2.resize(frame, (1020, 500))
            except:
                break

            results = model.predict(frame)
            #   print(results)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            #    print(px)
            list = []
            for index, row in px.iterrows():
                #        print(row)

                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                d = int(row[5])
                c = class_list[d]
                if 'car' in c:
                    list.append([x1, y1, x2, y2])
            bbox_idx = tracker.update(list)
            for bbox in bbox_idx:
                x3, y3, x4, y4, id1 = bbox
                cx = int(x3 + x4) // 2
                cy = int(y3 + y4) // 2

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)

                # display number of cars
                num_cars = len(bbox_idx)
                cv2.putText(frame, f'Number of Cars: {num_cars}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("RGB", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"Total Car Count : {tracker.id_count}")
        print(tracker.center_points)

        # cap.release()
        cv2.destroyAllWindows()

        # Render the template with the processed text
        return render_template('index.html', processed_text=processed_text)

    # If the request method is GET, render the template without processing
    return render_template('index.html')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x, y]
        print(colorsBGR)


if __name__ == '__main__':
    app.run(debug=True)
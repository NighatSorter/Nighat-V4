import cv2
import torch
from ultralytics import YOLO
import httpx
import logging
import warnings
import os
import asyncio
from pypylon import pylon  # Import Pylon for Basler camera

# Suppress the specific warning about weights_only
warnings.filterwarnings("ignore", message="It is possible to construct malicious pickle data")

# Set up logging to a file
logging.basicConfig(filename='./server_requests.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda:0")
print(device)

# Load the YOLO model
model = YOLO("512 - augmentation - V3.pt").to(device)

# Initialize Basler camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Initialize counters and sets for tracking
class_counters = {}
first_half_counters = {}
second_half_counters = {}
counted_ids = set()

middle_line_y = None
vertical_line_x = None

# Set to keep track of printed track_ids
printed_track_ids = []

async def send_request(send, class_id, track_ids, half_counter):
    async with httpx.AsyncClient() as client:
        try:
            print("=================> ", send)
            res = await client.get(f"http://192.168.5.118:8000/control-valve/?valve_id={send}")
            if res.status_code == 200:
                logging.info(f"Request was successful and valve should open now.")
            else:
                logging.info("Failed to control the valve and request failed when sent to RasPi")
            # Log the request
            logging.info(f"Sent as Valve Number: {send}, and Id Date is: {int(class_id)}, Truck Id {track_ids}, half counters: {half_counter}")
        except httpx.RequestError as e:
            print(f"==================> Request failed: {e}")
            logging.error(f"Request failed: {e}")

async def process_video():
    global middle_line_y, vertical_line_x  # Ensure variables can be updated

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                frame = converter.Convert(grab_result)
                frame = frame.GetArray()

                height, width = frame.shape[:2]
                print(width, height)
                scaling_factor = 512 / max(width, height)
                # Calculate the new dimensions
                new_w = int(width * scaling_factor)
                new_h = int(height * scaling_factor)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                if middle_line_y is None:
                    middle_line_y = new_h // 2
                    vertical_line_x = new_w // 2

                results = model.track(frame, persist=True, show=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id

                frame = cv2.line(frame, (0, middle_line_y), (new_w, middle_line_y), (255, 255, 255), 2)
                frame = cv2.line(frame, (vertical_line_x, 0), (vertical_line_x, new_h), (255, 255, 255), 2)

                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = map(int, box)
                    if track_ids is not None:
                        track_id = int(track_ids[i])
                        # Print the track_id only if it hasn't been printed before
                        if track_id not in [d["track_id"] for d in printed_track_ids]:
                            print(f"ID {track_id} - Class {int(class_id)}: {conf:.2f}")
                            printed_track_ids.append({
                                "track_id": track_id,
                                "sent_to_machine": False
                            })
                        label = f"ID {track_id} - Class {int(class_id)}: {conf:.2f}"
                    else:
                        label = f"Class {int(class_id)}: {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if class_id not in class_counters:
                        class_counters[class_id] = 0
                        first_half_counters[class_id] = 0
                        second_half_counters[class_id] = 0

                    if middle_line_y - 20 < center_y < middle_line_y + 20:
                        if center_x <= vertical_line_x:
                            send = str(3) + str(int(class_id) + 1)
                            first_half_counters[class_id] += 1

                            # Check if the track_id has already been printed ONCE
                            detected_object = next((d for d in printed_track_ids if d["track_id"] == track_id), None)
                            if detected_object and detected_object["sent_to_machine"] == False:
                                await send_request(send, class_id, track_ids, first_half_counters[class_id])
                                detected_object["sent_to_machine"] = True
                        else:
                            send = str(2) + str(int(class_id) + 1)
                            second_half_counters[class_id] += 1

                            # Check if the track_id has already been printed ONCE
                            detected_object = next((d for d in printed_track_ids if d["track_id"] == track_id), None)
                            if detected_object and detected_object["sent_to_machine"] == False:
                                await send_request(send, class_id, track_ids, second_half_counters[class_id])
                                detected_object["sent_to_machine"] = True

                    cv2.circle(frame, (center_x, center_y), radius=5, color=(255, 255, 255), thickness=-1)

                cv2.imshow("Live Stream", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            grab_result.Release()

    finally:
        camera.StopGrabbing()
        cv2.destroyAllWindows()

# Run the video processing function asynchronously
asyncio.run(process_video())

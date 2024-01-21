import math
import time
import cv2
import pygetwindow as gw
import torch
from PIL import Image
import numpy as np
import dxcam
import win32api, win32con
import keyboard
import sys
import tkinter as tk


def create_waiting_window(variables_to_return):
    preferred_classes = entry_preferred_classes.get()
    friendly_classes = entry_friendly_classes.get()
    min_confidence = entry_min_confidence.get()
    activation_key = entry_activation_key.get()
    game_name = entry_game_name.get()
    estimate = entry_estimate.get()
    estimation_control = entry_estimation_control.get()
    pretrained_weights = entry_pretrained_weights.get()
    yolo_installation_path = entry_yolo_installation_path.get()
    fine_tuned_model_path = entry_fine_tuned_model_path.get()


    preferred_classes = list(map(int, preferred_classes.split(',')))

    if friendly_classes == "":
        friendly_classes = []
    else:
        friendly_classes = list(map(int, friendly_classes.split(',')))
    min_confidence = float(min_confidence)
    estimate = estimate.lower() == 'true'
    estimation_control = int(estimation_control)
    pretrained_weights = pretrained_weights.lower() == 'true'


    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Preferred Classes: {preferred_classes}\n")
    result_text.insert(tk.END, f"Friendly Classes: {friendly_classes}\n")
    result_text.insert(tk.END, f"Min Confidence: {min_confidence}\n")
    result_text.insert(tk.END, f"Activation Key: {activation_key}\n")
    result_text.insert(tk.END, f"Game Name: {game_name}\n")
    result_text.insert(tk.END, f"Estimate: {estimate}\n")
    result_text.insert(tk.END, f"Estimation Offset: {estimation_control}\n")
    result_text.insert(tk.END, f"Pretrained Weights: {pretrained_weights}\n")
    result_text.insert(tk.END, f"YOLO Installation Path: {yolo_installation_path}\n")
    result_text.insert(tk.END, f"Fine-Tuned Model Path: {fine_tuned_model_path}\n")

    variables_to_return['preferred_classes'] = preferred_classes
    variables_to_return['friendly_classes'] = friendly_classes
    variables_to_return['min_confidence'] = min_confidence
    variables_to_return['activation_key'] = activation_key
    variables_to_return['game_name'] = game_name
    variables_to_return['estimate'] = estimate
    variables_to_return['estimation_control'] = estimation_control
    variables_to_return['pretrained_weights'] = pretrained_weights
    variables_to_return['yolo_installation_path'] = yolo_installation_path
    variables_to_return['fine_tuned_model_path'] = fine_tuned_model_path

window = tk.Tk()
window.title("Aim Assistant Parameters")

# Default parameter values
default_values = {
    "preferred_classes": "0",
    "friendly_classes": "",
    "min_confidence": "0.3",
    "activation_key": "ctrl",
    "game_name": "AssaultCube",
    "estimate": "True",
    "estimation_control": "-10",
    "pretrained_weights": "True",
    "yolo_installation_path": "C:\\Users\\stefa\\PycharmProjects\\Theses\\AutonomousDriving\\yolov5\\",
    "fine_tuned_model_path": "C:\\Users\\stefa\\PycharmProjects\\Theses\\AutonomousDriving\\yolov5\\runs\\train\\exp27\\weights\\best.pt",
}

entry_preferred_classes = tk.Entry(window, width=30)
entry_friendly_classes = tk.Entry(window, width=30)
entry_min_confidence = tk.Entry(window, width=30)
entry_activation_key = tk.Entry(window, width=30)
entry_game_name = tk.Entry(window, width=30)
entry_estimate = tk.Entry(window, width=30)
entry_estimation_control = tk.Entry(window, width=30)
entry_pretrained_weights = tk.Entry(window, width=30)
entry_yolo_installation_path = tk.Entry(window, width=30)
entry_fine_tuned_model_path = tk.Entry(window, width=30)

entry_preferred_classes.insert(tk.END, str(default_values["preferred_classes"]))
entry_friendly_classes.insert(tk.END, default_values["friendly_classes"])
entry_min_confidence.insert(tk.END, default_values["min_confidence"])
entry_activation_key.insert(tk.END, default_values["activation_key"])
entry_game_name.insert(tk.END, default_values["game_name"])
entry_estimate.insert(tk.END, default_values["estimate"])
entry_estimation_control.insert(tk.END, default_values["estimation_control"])
entry_pretrained_weights.insert(tk.END, default_values["pretrained_weights"])
entry_yolo_installation_path.insert(tk.END, default_values["yolo_installation_path"])
entry_fine_tuned_model_path.insert(tk.END, default_values["fine_tuned_model_path"])

entry_preferred_classes.grid(row=0, column=1, pady=5)
entry_friendly_classes.grid(row=1, column=1, pady=5)
entry_min_confidence.grid(row=2, column=1, pady=5)
entry_activation_key.grid(row=3, column=1, pady=5)
entry_game_name.grid(row=4, column=1, pady=5)
entry_estimate.grid(row=5, column=1, pady=5)
entry_estimation_control.grid(row=6, column=1, pady=5)
entry_pretrained_weights.grid(row=7, column=1, pady=5)
entry_yolo_installation_path.grid(row=8, column=1, pady=5)
entry_fine_tuned_model_path.grid(row=9, column=1, pady=5)

# Create labels for each entry
labels = ["Preferred Classes:", "Friendly Classes:", "Min Confidence:", "Activation Key:",
          "Game Name:", "Estimate:", "Estimation Offset:", "Pretrained Weights:",
          "YOLO Installation Path:", "Fine-Tuned Model Path:"]
for i, label_text in enumerate(labels):
    label = tk.Label(window, text=label_text)
    label.grid(row=i, column=0, pady=5, padx=10, sticky=tk.W)

# Create a big text field for displaying the result
result_text = tk.Text(window, height=10, width=50)
result_text.grid(row=11, column=0, columnspan=2, pady=10)

shared_variables = {}
button_cast_parameters = tk.Button(window, text="Cast Parameters", command=lambda: create_waiting_window(shared_variables))
button_cast_parameters.grid(row=10, column=0, pady=10, padx=5)

# Create a "Start" button
button_start_program = tk.Button(window, text="Start", command=window.destroy)
button_start_program.grid(row=10, column=1, pady=10, padx=5)

open_windows = gw.getAllTitles()
active_windows = ""
for title in open_windows:
    if title == "":
        continue
    active_windows = active_windows + "\n"+title

result_text.insert(tk.END, f"Active Windows (Game Name Parameter): {active_windows}")

window.mainloop()

print(shared_variables)

#----------Parameters-------------
preferred_classes = shared_variables['preferred_classes']
friendly_classes = shared_variables['friendly_classes']
min_confidence = shared_variables['min_confidence']
activation_key = shared_variables['activation_key']
game_name = shared_variables['game_name']
estimate = shared_variables['estimate']
estimation_control = shared_variables['estimation_control']
pretrained_weights = shared_variables['pretrained_weights']
yolo_installation_path = shared_variables['yolo_installation_path']
fine_tuned_model_path = shared_variables['fine_tuned_model_path']
#----------------------


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if pretrained_weights == True:
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
else:
    yolo = torch.hub.load(yolo_installation_path, 'custom', path=fine_tuned_model_path, source='local')
yolo = yolo.to(device)
yolo.eval()
enabled = False


#dummy_input = torch.randn(1, 3, 416, 416)
#onnx_path = "yolov5.onnx"
#torch.onnx.export(yolo, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])


def extract_bbox(frame,yolo_model,window_left,window_top,window_bottom,window_right):
    frame = np.ascontiguousarray(frame, dtype=np.uint8)
    pil_frame = Image.fromarray(frame.astype(np.uint8))
    results = yolo_model(pil_frame)

    results = results.xyxy[0]

    if pretrained_weights and not results.numel() == 0:
        target_class_ids = [0, 1]
        class_ids = results[:, -1]
        mask = torch.tensor([class_id in target_class_ids for class_id in class_ids])
        results = results[mask]

    confidences = []
    global_centers = []
    local_centers = []
    detected_classes = []
    global_distances = []
    center_widths = []

    for pred in results:
        x1, y1, x2, y2, conf, class_idx = pred[:6]
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        conf = conf.item()
        class_idx = int(class_idx.item())

        local_center_x = (x1 + x2) / 2
        local_center_y = (y1 + y2) / 2
        local_centers.append((local_center_x,local_center_y))
        global_center_x = local_center_x + window_left
        global_center_y = local_center_y + window_top
        confidences.append(conf)
        detected_classes.append(class_idx)
        global_centers.append((global_center_x,global_center_y))
        original_x, original_y = win32api.GetCursorPos()
        global_distances.append(math.sqrt((global_center_x - original_x)**2 + (global_center_y - original_y )**2))
        center_widths.append(local_center_y-y2)

        thickness = 2
        if class_idx in friendly_classes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 8, 255), thickness)
            cv2.putText(frame, f"{'friendly'}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 8, 255), 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (156, 0, 0), thickness)
            cv2.putText(frame, f"{'enemy'}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (156, 0, 0), 1)

    if len(global_distances) > 0:
        detected_data = list(zip(global_distances, detected_classes, global_centers, confidences, local_centers, center_widths))
        detected_data_classes = sorted(detected_data, key=lambda x: x[0])
        global_distances, detected_classes, global_centers, confidences, local_centers,center_widths = zip(*detected_data_classes)

        height, width = frame.shape[:2]
        bottom_middle_x = width // 2
        bottom_middle_y = height

        best_center = 0
        minimal_distance = 10000000000000000000000
        best_idx = 0
        found_best = False
        for i in range(len(global_distances)):
            if detected_classes[i] in preferred_classes:
                if confidences[i] >= min_confidence:
                    if global_distances[i] <= minimal_distance:
                        best_center = global_centers[i]
                        minimal_distance = global_distances[i]
                        best_idx = i
                        found_best = True

        if found_best:
            cv2.line(frame, (int(bottom_middle_x), int(bottom_middle_y)), (int(local_centers[best_idx][0]),int(local_centers[best_idx][1])), (156, 0, 0), 3)

        if enabled:
            original_x, original_y = win32api.GetCursorPos()
            if not best_center == 0:
                if estimate:
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(best_center[0] - original_x),int(best_center[1] - original_y) + (int(center_widths[best_idx])-int(estimation_control)), 0, 0)
                else:
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(best_center[0]-original_x), int(best_center[1]-original_y), 0, 0)
    return frame

game_window = gw.getWindowsWithTitle(game_name)[0]
game_window.activate()

left, top, right, bottom, width, height = game_window.left, game_window.top, game_window.right,game_window.bottom, game_window.width, game_window.height

camera = dxcam.create()
loop_time = time.time()
camera.start(target_fps=144)
while True:
    try:
        if keyboard.is_pressed(activation_key):
            enabled = True
        else:
            enabled = False

        left, top, right, bottom, width, height = game_window.left, game_window.top, game_window.right, game_window.bottom, game_window.width, game_window.height
        image = camera.grab(region=(left, top, right, bottom))
        image = extract_bbox(image, yolo, left, top, bottom, right)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("Python Window", image)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

        print('FPS {}'.format(1 / (time.time() - loop_time)))
        loop_time = time.time()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno': exc_traceback.tb_lineno,
            'name': exc_traceback.tb_frame.f_code.co_name,
            'type': exc_type.__name__,
            'message': str(exc_value)
        }

        print(
            f"Exception at line {traceback_details['lineno']}: {traceback_details['type']}: {traceback_details['message']}")
camera.stop()
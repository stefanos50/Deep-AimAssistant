# Deep-AimAssistant

This is a generic aim assistant application that employs AI (Object Detection) and automatically detects and points detected objects in real-time applications without interfering with the actual executable or files of the target application. `The code was written specifically for educational purposes on AI techniques, and that is why the fine-tuned weights of the trained models are not included`. The code can still be applied with some limitations by employing the default pre-trained models that YOLOv5 provides on the COCO dataset, but use it at your own risk. This is the extension of a [previous repository](https://github.com/stefanos50/Real-Time-Object-Detection-In-Games) where Python plays an endless runner game also based on object detection.

# How it Works

The approach detects objects based on the YOLOv5 (medium size) model at around 33FPS in real-time with an RTX 4090 by capturing screenshots of a target window on the screen, which can be a real-time application. Based on the center of the detected box location that is closer to the current mouse location (to smooth the selection), after converting the coordinates to the full resolution scale, the mouse is automatically moved by the Windows API. If the target location is not the center of the bounding box, then the location can be dynamically modified by the bounding box properties and a pre-selected offset. Additional parameters (target class selection, team color selection, confidence threshold, activation key, etc.) can be modified by a window that provides a UI at the start of the execution of the script.

# ToDo 

* Add Nvidia TensorRT compiler to improve the real-time performance.

# Execution Example

The code was tested in Assault Cube vs. Bots. Please, as already stated, if you use this code, use it only in offline applications or AI opponents.

https://github.com/stefanos50/Deep-AimAssistant/assets/36155283/9d039670-a7d0-45d5-8d75-64869a9a038a


import torch

weights_path = './models/non/best_93.pt'
output_path = './models/exported/traffic_sign_detection.pt'

# Define input size
input_height = 224
input_width = 224

# Load the pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Export the model
torch.onnx.export(model, torch.zeros(1, 3, input_height, input_width), output_path, verbose=False)
# Save the exported model in the desired format
model.model.save(output_path)
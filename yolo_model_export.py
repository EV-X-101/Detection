import torch

weights_path = './models/non/best_93.pt'
output_path = './models/exported/traffic_sign_detection.pt'

# Load the pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

# Export the model
torch.onnx.export(model.model, torch.zeros(1, 3, model.model.hyp['height'], model.model.hyp['width']),
                  output_path, verbose=False)

# Save the exported model in the desired format
model.model.save(output_path)
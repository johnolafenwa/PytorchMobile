import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2(pretrained=True)

model.eval()
input_tensor = torch.rand(1,3,224,224)

script_model = torch.jit.trace(model,input_tensor)
script_model.save("mobilenet-v2.pt")
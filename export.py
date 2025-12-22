from ViT import ViT
import torch
import torch.onnx

model = ViT(img_size=224, patch_size=32, embed_dim=512, num_heads=8, num_blocks=12, num_classes=2, bn=512)
model.load('saved11.pth') 
model.eval()

example = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    example,
    "model11.onnx",
    input_names=["input"],
    output_names=["output"]
)
from ViT import ViT
import torch
from torchvision.transforms import v2
from PIL import Image
import sys, json, struct, base64
from io import BytesIO

def getMessage():
    raw_length = sys.stdin.buffer.read(4)
    if not raw_length: return None

    length = struct.unpack('@I', raw_length)[0]
    message = sys.stdin.buffer.read(length).decode('utf-8')

    try:
        data = json.loads(message)
        image = base64.b64decode(data['image'])
        return Image.open(BytesIO(image))
    
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        return None
        
def encodeMessage(messageContent):
    labels = {0: 'Fake', 1: 'Real'}

    if "Warning" in messageContent:
        msg = f'Warning, Low Confidence: This image is {labels[messageContent[-1]]}'
    else:
        msg = f'This image is {labels[messageContent[-1]]}'

    encodedContent = json.dumps(msg, separators=(',', ':')).encode('utf-8')
    encodedLength = struct.pack('I', len(encodedContent))
    return {'length': encodedLength, 'content': encodedContent}

def sendMessage(encodedMessage):
    sys.stdout.buffer.write(encodedMessage['length'])
    sys.stdout.buffer.write(encodedMessage['content'])
    sys.stdout.buffer.flush()

transform = v2.Compose([
    v2.ToImage(),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = ViT(img_size=224, patch_size=32, embed_dim=512, num_heads=8, num_blocks=12, num_classes=2, bn=512)
model.load('saved12.pth')
model.eval()

while True:
    receivedMessage = getMessage()
    if receivedMessage:
        with torch.inference_mode():
            run = model(transform(receivedMessage).unsqueeze(dim=0))

        probs = torch.softmax(run[0], dim=0)
        prd = torch.argmax(probs, dim=0)

        if max(probs) < 0.65:
            sendMessage(encodeMessage(["Warning", prd.item()]))
        else:
            sendMessage(encodeMessage([prd.item()]))
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
Image.MAX_IMAGE_PIXELS = None

class ImgResize(object):
    def __call__(self, img: Image, output_size=(224,224)):
        image = img

        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        target_width, target_height = output_size
        if target_width / target_height > aspect_ratio:
            target_width = int(target_height * aspect_ratio)
        else:
            target_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image = image.resize((target_width, target_height), Image.LANCZOS)

        return resized_image

class PadToSquare:
    def __call__(self, img: Image):

        width, height = img.size
 
        target_size = max(width, height)

        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
  
        x_offset = (target_size - width) // 2
        y_offset = (target_size - height) // 2
        
        new_img.paste(img, (x_offset, y_offset))
        
        return new_img

BATCH_SIZE = 512

transform = transforms.Compose([
    ImgResize(),
    PadToSquare(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=[0.3530, 0.3320, 0.3074],
                         std=[0.3301, 0.3119, 0.3061])
])

dataset=ImageFolder("archive", transform=transform)
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=0)

train_data = DataLoader(dataset=train_set,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

test_data = DataLoader(dataset=test_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False)
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

dataset_dir = '../../jrdb_train/cvgl/group/jrdb/data/train_dataset/'
# image_address = dataset_dir + 'images/image_0/cubberly-auditorium-2019-04-22_0/000018.jpg'
# save_address = '../some-checking/image_0______________cubberly-auditorium-2019-04-22_0/'
image_address = dataset_dir + 'images/image_stitched/clark-center-2019-02-28_1/000000.jpg'
save_address = '../some-checking/stitched__________clark-center-2019-02-28_1/'

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
print(model)
model.eval()
model = model.cuda()

# img = Image.open(image_address).resize((800, 600)).convert('RGB')
img = Image.open(image_address).convert('RGB')
img_tens = transform(img).unsqueeze(0).cuda()

with torch.no_grad():
    output = model(img_tens)


im2 = img.copy()
drw = ImageDraw.Draw(im2)
pred_logits=output['pred_logits'][0][:, :len(CLASSES)]
pred_boxes=output['pred_boxes'][0]

max_output = pred_logits.softmax(-1).max(-1)    # we can filter here, given a threshold
topk = max_output.values.topk(60)

pred_logits = pred_logits[topk.indices]
pred_boxes = pred_boxes[topk.indices]
print(pred_logits.shape)

for logits, box in zip(pred_logits, pred_boxes):
    cls = logits.argmax()
    if cls >= len(CLASSES):
        continue

    label = CLASSES[cls]
    print(label)
    # box = box.cpu() * torch.Tensor([800, 600, 800, 600])
    box = box.cpu() * torch.Tensor([3760, 480, 3760, 480])
    x, y, w, h = box
    x0, x1 = x-w//2, x+w//2
    y0, y1 = y-h//2, y+h//2
    drw.rectangle([x0, y0, x1, y1], outline='red', width=2)
    drw.text((x, y), label, fill='white')

# im2.show()
im2.save(save_address + "000018_output.jpg", "JPEG")
import cv2
import torch
import numpy as np
import albumentations as A

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

def get_image(img):
    
    img, _, _ = letterbox(img)

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img

class YoloEngine:

    def __init__(self, model_path, device='', conf_thres=0.12, iou_thres=0.45):
        self.device = select_device(str(device))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Load model
        self.model = attempt_load(model_path, map_location=self.device)  # load FP32 model
        self.model.eval()

        # self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        # print(self.names)
        self.names = ['Computer monitor', 'Chair', 'Computer keyboard', 'Snack', 'Human face', 
                        'Pillow', 'Apple', 'Watch', 'Hat', 'Microphone', 'Bidet', 'Sunglasses', 
                        'Clock', 'Shrimp', 'Sun hat', 'Guitar', 'Microwave oven', 'Shirt', 
                        'Book', 'Coffee', 'Ball', 'Dog', 'Headphones', 'Bird', 'Belt', 
                        'Cucumber', 'Door', 'Balloon', 'Beer', 'Sandwich', 'Suitcase', 
                        'Vase', 'Cat', 'Desk', 'Goldfish', 'Bathtub', 'Computer mouse', 
                        'Sink', 'Chopsticks', 'Box', 'Coffee cup', 'Egg', 'Crab', 
                        'Pancake', 'Washing machine', 'Bicycle', 'Laptop']

        # img = torch.zeros(1, 3, 640, 640).to(self.device)
        # self.model(img)
        

    def detect(self, image):

        # preprocess input
        img = get_image(image).to(self.device).type(torch.float32)

        #predicting
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        if pred.shape[0] == 0:
            return None
        
        # visualize boxes
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()

        for x1, y1, x2, y2, prob, name_class in pred.numpy():
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
            cv2.putText(image, self.names[int(name_class)], (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)

        cv2.imshow("test", image)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # handle output
        classes = pred[:,-1].to('cpu').numpy().astype(int)
        out_classes = [self.names[e] for e in classes]

        return list(set(out_classes))


class CatEngine:
    
    def __init__(self, device=''):
        self.engine_detect = YoloEngine(model_path='best.pt', device=device)
    
    def predict(self, image_path):
        image = cv2.imread(image_path)

        # predict with model detection
        out_detect = self.engine_detect.detect(image)

        return out_detect

if __name__ == '__main__':
    # Load data
    image_path = 'test_images/8.4c-ke-de-ipad-ban-lam-viec.jpg'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cat_engine = CatEngine(device=device)

    preds = cat_engine.predict(image_path)
    print("======================")
    print("Detected objects: ", preds)
    print("======================")

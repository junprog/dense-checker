import torch
from torchvision import transforms
import numpy as np

from models.vgg import vgg19
from models.mobilenet import mobilenet_v2

class Counter(object):
    def __init__(self, model='vgg', model_path='data/ucf_vgg_best_model.pth'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        if model == 'vgg':
            self.model = vgg19()
        elif model == 'mobilenet':
            self.model = mobilenet_v2()
        
        # load pre-trained model
        self.model.load_state_dict(torch.load(model_path, self.device))
        # model to GPU or CPU
        self.model.to(self.device)

    @torch.no_grad()
    def regression(self, img):
        self.model.eval()

        img = self.trans(img).unsqueeze_(0)
        img = img.to(self.device).detach()
        out = self.model(img)

        out = out.to('cpu').detach().numpy().copy()
        dense_map = np.squeeze(out)

        count = int(round(dense_map.sum()))

        return dense_map, count
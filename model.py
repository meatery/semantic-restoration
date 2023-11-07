import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from nets.model import generator
from utils.utils import (cvtColor, postprocess_output, preprocess_input,
                         resize_image, show_config)


class OURMODEL(object):
    _defaults = {

        "model_path"        : 'model_data/G.pth',

        "input_shape"       : [256, 512],

        "letterbox_image"   : True,

        "cuda"              : False,
    }


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()
        
        show_config(**self._defaults)

    def generate(self):

        self.net    = generator().eval()

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()


    def detect_image(self, image):

        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = pr.permute(1, 2, 0).cpu().numpy()
            if nw is not None:
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            
        image = postprocess_output(pr)
        image = Image.fromarray(np.uint8(image))

        return image

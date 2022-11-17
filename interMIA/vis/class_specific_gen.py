import os
import numpy as np
import torch
import nibabel as nib
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import models



def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def preprocess_image(img):
    im_as_arr = np.float32(img)

    norm = normalize(im_as_arr)
    ten = torch.from_numpy(norm).float()
    ten = ten.unsqueeze(0)
    ten_var = Variable(ten, requires_grad=True)
    return ten_var

def save_img(im, path):
    vol = nib.Nifti1Image(im[0], affine=np.eye(4))
    nib.save(vol, path)



class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """

    def __init__(self, model, target_class):
        self.model = model
        self.model.eval()

        self.target_class = target_class

        self.created_image = np.random.rand(2, 32, 32, 32)
        self.bu = 0

        if not os.path.exists('data/vis/class_'+str(self.target_class)):
            os.makedirs('data/vis/class_'+str(self.target_class))

    def generate(self, iterations=1500):
        """Generates class specific image
        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})
        Returns:
            np.ndarray -- Final maximally activated class image
        """
        initial_learning_rate = 0.01
        for i in range(1, iterations):
            # Process image and return variable
            self.proc_img = preprocess_image(self.created_image)
            optimizer = SGD([self.proc_img], lr=initial_learning_rate)

            out = self.model(self.proc_img)
            class_loss = -out[0, self.target_class]


            if i % 10 == 0 or i == iterations-1:
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.data.numpy()))

            self.model.zero_grad()
            class_loss.backward()
            optimizer.step()
            self.created_img = self.proc_img

            if i % 10 == 0 or i == iterations-1:
                # Save image
                im_path = 'data/vis/class_' + \
                    str(self.target_class)+'/c_' + \
                    str(self.target_class)+'_'+'iter_'+str(i)+'.nii.gz'
                save_img(self.created_image, im_path)

        return self.proc_img




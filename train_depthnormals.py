import os
import sys
import numpy as np

from PIL import Image

import net
from utils import depth_montage, normals_montage
#from keras.optimizers import SGD

def main():
    # location of depth module, config and parameters

    model_name = 'depthnormals_nyud_alexnet'
    #model_name = 'depthnormals_nyud_vgg'

#    module_fn = 'models/iccv15/%s.py' % model_name
#    config_fn = 'models/iccv15/%s.conf' % model_name
#    params_dir = 'weights/iccv15/%s' % model_name

    module_fn = 'models/iccv15/depthnormals_nyud_alexnet.py' #% model_name
    config_fn = 'models/iccv15/depthnormals_nyud_alexnet.conf' #% model_name
    params_dir = 'weights/iccv15/depthnormals_nyud_alexnet' #% model_name
    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    # demo image
    rgb = Image.open('thyroid.jpg')
    rgb = rgb.resize((320, 240), Image.BICUBIC)
    #rergb = Image.fromarray(rgb)
    rgb.save("thyroid.jpg")

    # build depth inference function and run
    rgb_imgs = np.asarray(rgb).reshape((1, 240, 320, 3))
    (pred_depths, pred_normals) = machine.infer_depth_and_normals(rgb_imgs)

    # save prediction
    depth_img_np = depth_montage(pred_depths)
    depth_img = Image.fromarray((255*depth_img_np).astype(np.uint8))
    depth_img.save('demo_depth_prediction_thyroid.png')

    normals_img_np = normals_montage(pred_normals)
    normals_img = Image.fromarray((255*normals_img_np).astype(np.uint8))
    normals_img.save('demo_normals_prediction_thyroid.png')
def machine2model():
    module_fn = 'models/iccv15/depthnormals_nyud_alexnet.py' #% model_name
    config_fn = 'models/iccv15/depthnormals_nyud_alexnet.conf' #% model_name
    params_dir = 'weights/iccv15/depthnormals_nyud_alexnet' #% model_name
    trainMachine = net.create_machine(module_fn, config_fn, params_dir)
    #model = trainMachine.define_machine()
    print trainMachine
    
    return trainMachine
def traindepth(data = None):
    #traindata, testdata = data('train','test')
    module_fn = 'models/iccv15/depthnormals_nyud_alexnet.py' #% model_name
    config_fn = 'models/iccv15/depthnormals_nyud_alexnet.conf' #% model_name
    params_dir = 'weights/iccv15/depthnormals_nyud_alexnet' #% model_name
    #trainMachine = net.create_machine(module_fn, config_fn, params_dir)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model = Sequential()
    model = machine2model()
    print "======>>", model
    loss = machine.define_depths_cost(self, pred, y0, m0)
    #model.compile(loss=loss,optimizer = sgd, metrics=['accuracy'])
    #model.fit(x_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, validation_data = (x_test, y_test), shuffle = True)

if __name__ == '__main__':
    #main()
    traindepth('')


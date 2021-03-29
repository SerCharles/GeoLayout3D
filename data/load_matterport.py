import h5py
import numpy as np
import os
import torch
import scipy.io as sio
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import PIL
import cv2


class MatterPortDataSet(Dataset):
    def __init__(self, base_dir, the_type):
        '''
        description: init the dataset
        parameter: the base dir of the dataset, the type(training, validation, testing)
        return:empty
        '''
        self.setTransform()

        self.base_dir = base_dir 
        self.type = the_type
        self.mat_path = os.path.join(base_dir, the_type, the_type + '.mat')
        self.depth_filenames = []
        self.image_filenames = []
        self.init_label_filenames = []
        self.layout_depth_filenames = []
        self.layout_seg_filenames = []
        self.points = []
        self.intrinsics = []
        self.faces = []
        self.params = []
        self.norms_x = [] 
        self.norms_y = [] 
        self.norms_z = [] 

        self.boundarys = [] 
        self.radiuss = [] 
    

        with h5py.File(self.mat_path, 'r') as f:
            data = f['data']
            
            images = data['image'][:]
            intrinsics_matrixs = data['intrinsics_matrix'][:]
            self.length = len(images)

            if not the_type == 'testing':
                depths = data['depth'][:]
                images = data['image'][:]
                init_labels = data['init_label'][:]
                intrinsics_matrixs = data['intrinsics_matrix'][:]
                layout_depths = data['layout_depth'][:]
                layout_segs = data['layout_seg'][:]
                models = data['model'][:]
                points = data['point'][:]

            
            for i in range(self.length):
                image_id = f[images[i][0]]
                the_string = ''
                for item in image_id: 
                    the_string += chr(item[0])
                self.image_filenames.append(the_string)

                the_intrinsic = f[intrinsics_matrixs[i][0]]
                the_intrinsic = np.array(the_intrinsic)
                self.intrinsics.append(the_intrinsic)


                if not the_type == 'testing':
    
                    depth_id = f[depths[i][0]]
                    init_label_id = f[init_labels[i][0]]
                    layout_depth_id = f[layout_depths[i][0]]
                    layout_seg_id = f[layout_segs[i][0]]
                    the_model = f[models[i][0]]
                    the_point = f[points[i][0]]

                    the_point = np.array(the_point)
                    the_model_faces = the_model['face']
                    the_model_params = the_model['params']


                    faces = []
                    params = []
                    for j in range(len(the_model_faces)):
                        face = f[the_model_faces[j][0]][0][0]
                        param = f[the_model_params[j][0]][0][0]
                        faces.append(face)  
                        params.append(param)
                    self.faces.append(faces) 
                    self.params.append(params) 
                
                    self.points.append(the_point)

                    the_string = ''
                    for item in depth_id: 
                        the_string += chr(item[0])
                    self.depth_filenames.append(the_string)


                    the_string = ''
                    for item in init_label_id: 
                        the_string += chr(item[0])
                    self.init_label_filenames.append(the_string)

                    the_string = ''
                    for item in layout_depth_id: 
                        the_string += chr(item[0])
                    self.layout_depth_filenames.append(the_string)

                    the_string = ''
                    for item in layout_seg_id: 
                        the_string += chr(item[0])
                    self.layout_seg_filenames.append(the_string)

        self.depths = [] 
        self.images = [] 
        self.init_labels = [] 
        self.layout_depths = []
        self.layout_segs = []
        for i in range(self.length):

            image_name = os.path.join(self.base_dir, self.type, 'image', self.image_filenames[i])
            image = Image.open(image_name)
            self.images.append(image)
            if not the_type == 'testing':
                base_name = self.depth_filenames[i][:-4]
                depth_name = os.path.join(self.base_dir, self.type, 'depth', self.depth_filenames[i])
                init_label_name = os.path.join(self.base_dir, self.type, 'init_label', self.init_label_filenames[i])
                layout_depth_name = os.path.join(self.base_dir, self.type, 'layout_depth', self.layout_depth_filenames[i])
                layout_seg_name = os.path.join(self.base_dir, self.type, 'layout_seg', self.layout_seg_filenames[i])
                nx_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_nx.png')
                ny_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_ny.png')
                nz_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_nz.png')
                boundary_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_boundary.png')
                radius_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_radius.png')

                depth = Image.open(depth_name).convert('I')
                init_label = Image.open(init_label_name).convert('I')
                layout_depth = Image.open(layout_depth_name).convert('I')
                layout_seg = Image.open(layout_seg_name).convert('I')
                nx = Image.open(nx_name).convert('I')
                ny = Image.open(ny_name).convert('I')
                nz = Image.open(nz_name).convert('I')
                boundary = Image.open(boundary_name).convert('I')
                radius = Image.open(radius_name).convert('I')

                self.depths.append(depth)
                self.init_labels.append(init_label)
                self.layout_depths.append(layout_depth)
                self.layout_segs.append(layout_seg)
                self.norms_x.append(nx)  
                self.norms_y.append(ny)  
                self.norms_z.append(nz)  

                self.boundarys.append(boundary) 
                self.radiuss.append(radius)  

    def setTransform(self):
        '''
        description: set the transformation of the input picture
        parameter: empty
        return: empty
        '''
        self.transform_picture = transforms.Compose([
                                transforms.RandomResizedCrop([304, 228]),
                                transforms.ToTensor(),
                                transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, )])
        self.transform_depth = transforms.Compose([transforms.Resize([152, 114]), transforms.ToTensor()])
        self.transform_seg = transforms.Compose([transforms.Resize([152, 114], interpolation = PIL.Image.NEAREST), transforms.ToTensor()])
        #self.transform_seg = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, i):
        '''
        description: get one part of the item
        parameter: the index 
        return: the data
        '''
        if self.type == 'testing':
            image = self.transform_picture(self.images[i])
            return image, self.intrinsics[i]
        else:
            depth = self.transform_depth(self.depths[i])
            image = self.transform_picture(self.images[i])
            init_label = self.transform_seg(self.init_labels[i])
            layout_depth = self.transform_depth(self.layout_depths[i])
            layout_seg = self.transform_seg(self.layout_segs[i])
            nx = self.transform_depth(self.norms_x[i])
            ny = self.transform_depth(self.norms_y[i])
            nz = self.transform_depth(self.norms_z[i])
            nx = nx.resize(nx.size()[1], nx.size()[2])
            ny = ny.resize(ny.size()[1], ny.size()[2])
            nz = nz.resize(nz.size()[1], nz.size()[2])
            norm = torch.stack((nx, ny, nz))
            return depth, image, init_label, layout_depth, layout_seg, \
            self.faces[i], self.intrinsics[i], norm
 
    def __len__(self):
        '''
        description: get the length of the dataset
        parameter: empty
        return: the length
        '''
        return self.length


'''
a = MatterPortDataSet('E:\\dataset\\geolayout', 'validation')
print('length:', a.__len__())
depth, image, init_label, layout_depth, layout_seg, face, intrinsic, norm = a.__getitem__(0)
print('depth:', depth, depth.size())
print('image:', image, image.size())
print('init_label:', init_label, init_label.size())
print('layout_depth:', layout_depth, layout_depth.size())
print('layout_seg:', layout_seg, layout_seg.size())
print('face:', face, len(face))
print('intrinsic:', intrinsic, intrinsic.shape)
print('norm:', norm, norm.size())


b = MatterPortDataSet('E:\\dataset\\geolayout', 'testing')
print('length:', b.__len__())
image, intrinsic = b.__getitem__(10)
print('image:', image, image.size())
print('intrinsic:', intrinsic, intrinsic.shape)
'''
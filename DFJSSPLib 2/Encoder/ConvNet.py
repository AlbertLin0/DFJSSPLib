import os.path
import re
import sys

import numpy as np
from scipy.fft import dst, idst
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from common import *
from torch.optim.lr_scheduler import StepLR


def gradient_angles_calculator(shape, center, u, depth, r):
    """
    calculate the x, y angle of gradient in each pixel
    :param shape: [HxW] the size of the captured image, in pixel
    :param center: [Rx, Ry] the sphere center 2d, in mm
    :param u: the ratio with the unit of mm/pixel
    :param depth: depth of the sphere, the distance between the cut plane and the bottom, in mm
    :param r: radius of the sphere, in mm
    :return: the x-axis angel and gradient matrix,  the y-axis angle and gradient matrix
    """
    H, W = shape
    angle_x = np.zeros([H, W])
    Gx = np.zeros([H, W])
    angle_y = np.zeros([H, W])
    Gy = np.zeros([H, W])

    r_ = np.sqrt(depth * (2 * r - depth))  # the radius of the cut circle in mm

    for i in range(H):
        for j in range(W):
            Nx = u * j - center[0]
            Ny = u * i - center[1]
            if Nx ** 2 + Ny ** 2 > r_ ** 2:
                continue

            Nz = np.sqrt(r ** 2 - Nx ** 2 - Ny ** 2)

            Gx[i, j] = Nx / Nz
            angle_x[i, j] = np.arctan(Nx / Nz)
            Gy[i, j] = Ny / Nz
            angle_y[i, j] = np.arctan(Ny / Nz)

    return angle_x, Gx, angle_y, Gy


def poisson_solver(gx, gy, boundary_image):
    """
    gx, gy are gradients, gx = nx/nz, gy = ny/nz while nz != 0
    according to https://web.media.mit.edu/~raskar/photo/code.pdf

    :param gx: gradient matrix on x of the image
    :param gy: gradient matrix on y of the image
    :param boundary_image: gray image
    :return: height map
    """
    pi = np.pi
    H, W = boundary_image.shape
    gxx = np.zeros([H, W])
    gyy = np.zeros([H, W])

    for i in range(0, H - 1):
        for j in range(0, W - 1):
            gyy[i + 1, j] = gy[i + 1, j] - gy[i, j]
            gxx[i, j + 1] = gx[i, j + 1] - gx[i, j]

    f = gxx + gyy
    boundary_image[1:-1, 1:-1] = 0.0
    f_bp = np.zeros([H, W])
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            f_bp[i, j] = -4 * boundary_image[i, j] + boundary_image[i, j + 1] + boundary_image[i, j - 1] + \
                         boundary_image[i - 1, j] + boundary_image[i + 1, j]

    f1 = f - f_bp.reshape([H, W])

    f2 = f1[1:H - 1, 1:W - 1]
    tt = dst(f2)
    f2sin = dst(tt.T).T

    x, y = np.meshgrid(range(1, W - 1), range(1, H - 1))
    denom = (2 * np.cos(pi * x / (W - 1)) - 2) + (2 * np.cos(pi * y / (H - 1)) - 2)
    f3 = f2sin / denom
    tt = idst(f3)
    img_tt = idst(tt.T).T

    img_direct = boundary_image
    img_direct[1:H - 1, 1:W - 1] = 0
    img_direct[1:H - 1, 1:W - 1] = img_tt

    return img_direct


def images_height_reconstruct(gxs, gys, boundary_images):
    """

    :param gxs: NxHxW
    :param gys: NxHxW
    :param boundary_images: NxHxWx3
    :return: reconstruct height maps
    """
    height_maps = torch.tensor([]).cuda()

    return height_maps


def config_parser(config):
    """

    :param config: 每张采集图片对应的信息，"FrameNum:[] CH1:[] CH2:[] ··· Coordinate_X: [] Coordinate_Y: [] Depth: []"
    :return: frame_id: frameNum 每一帧图的编号  x:球心的x坐标 y:球心的y坐标 depth:球的深度
    """
    digits = re.findall("[-+]?(?:\d*\.\d+|\d+)", config)
    frame_id = digits[0]
    x = eval(digits[-3])
    y = eval(digits[-2])
    depth = eval(digits[-1])
    return frame_id, x, y, depth


def generate_angles(config_path, data_path, shape, ratio, radius):
    """
    生成球的角度图标签，x、y标签分别存储在data/angle/x、data/angle/y文件夹下
    :param shape: 采集图片的尺寸
    :param data_path: 角度图存储的路径
    :param radius: 球体的半径
    :param ratio: 毫米像素比值  mm/pixel
    :param config_path: 清洗之后属性文件的位置
    :return: 球每个位置的标准角度图，存储路径为 data_path/angle/[frame_id]_x, [frame_id]_y
    """
    angle_x_path = data_path + '/angle/x'
    if not os.path.exists(angle_x_path):
        os.makedirs(angle_x_path)

    angle_y_path = data_path + '/angle/y'
    if not os.path.exists(angle_y_path):
        os.makedirs(angle_y_path)

    with open(config_path, 'r') as f:
        for config in f.readlines():
            frame_id, x, y, depth = config_parser(config)
            angle_x, _, angle_y, _ = gradient_angles_calculator(shape, [x, y], ratio, depth, radius)
            np.save(angle_x_path + '/' + frame_id + ".npy", angle_x)
            np.save(angle_y_path + '/' + frame_id + ".npy", angle_y)


IMAGE_SHAPE = [640, 480]
MM_PIXEL_RATIO = 0.038


def angle_maps_generator(model_list, data_path, radius_map):
    """
    生成所需模型的角度图
    :param model_list:
    :param data_path:
    :param radius_map:
    :return:
    """
    for model in model_list:
        path1 = data_path + '/' + model
        for sub_dir in os.listdir(path1):
            inner_path = path1 + '/' + sub_dir
            if os.path.isfile(inner_path):
                continue

            clean_info_list = inner_path + '/clean_info.txt'
            generate_angles(clean_info_list, inner_path, IMAGE_SHAPE, MM_PIXEL_RATIO, radius_map[model])


class BaselineDataset(Dataset):
    """
    数据集划分与TactileDataset保持一致，将label替换为了angle_x, angle_y
    对于图片的处理，依据原文中 原图片除以初始图片，并添加了两通道的x，y位置信息
    """

    def __init__(self, split, data_path, size_img=(320, 240), size_label=(320, 240), seed=777):
        self.data_path = data_path
        self.seed = seed
        self.split = split

        self.size_img = size_img
        self.size_label = size_label

        self.data_info = 'sphere_inf.txt' if self.split != 'test' else 'model_inf_new.txt'
        self.model_list = self.get_model_list()
        self.data_list = self.read_file()
        self.split_list = self.split_dataset()
        self.original = Image.open('./data/sphere1/2021_10_20_134628/image/1.png').convert('RGB')

    def __getitem__(self, index):
        img = Image.open(self.split_list[index][0]).convert('RGB')
        angle_x = Image.fromarray(np.load(self.split_list[index][1], allow_pickle=True))
        angle_y = Image.fromarray(np.load(self.split_list[index][2], allow_pickle=True))
        depth = Image.fromarray(np.load(self.split_list[index][3], allow_pickle=True)) if self.split=='test' else torch.zeros(self.size_label)

        img, angle_x, angle_y = self.img_transform(img, angle_x, angle_y)
        return {'img': img, 'angle_x': angle_x, 'angle_y': angle_y, 'depth': depth}

    def __len__(self):
        return len(self.split_list)

    def get_model_list(self):
        model_list = []
        with open(self.data_path + "/" + self.data_info, "r", encoding='utf-8') as f1:
            model_list += [line1.split(" ")[0] for line1 in f1.readlines()]

        return model_list

    def split_dataset(self):
        train_and_val, test = train_test_split(self.data_list, test_size=0.2, shuffle=True, random_state=self.seed)
        train_data, val = train_test_split(train_and_val, test_size=0.15, shuffle=True, random_state=self.seed)

        if self.split == 'train':
            return train_data
        elif self.split == 'val':
            return val
        else:
            return test

    def read_file(self):
        img_list = []
        angle_x_list = []
        angle_y_list = []
        depth_list = []

        for model in self.model_list:
            path1 = self.data_path + '/' + model
            for dir1 in os.listdir(path1):
                inner_path = path1 + '/' + str(dir1)
                if os.path.isfile(inner_path):
                    continue

                img_path = inner_path + "/image"
                depth_path = inner_path + "/depth"

                angle_x_path = inner_path + "/angle/x"
                angle_y_path = inner_path + "/angle/y"

                filter_path = angle_x_path if self.split != 'test' else depth_path

                for name in os.listdir(filter_path):
                    img_name = name.replace("npy", "png")
                    if img_name in os.listdir(img_path):
                        img_list += [img_path + '/' + str(img_name)]
                        angle_x_list += [angle_x_path + '/' + str(name)]
                        angle_y_list += [angle_y_path + '/' + str(name)]
                        depth_list += [depth_path + '/' + str(name)]

        return list(zip(img_list, angle_x_list, angle_y_list, depth_list))

    def img_transform(self, img, angle_x, angle_y):
        transform_img = transforms.Compose(
            [
                transforms.Resize(self.size_img),
                transforms.ToTensor(),
            ]
        )

        transform_label = transforms.Compose(
            [
                transforms.Resize(self.size_label),
                transforms.ToTensor(),
            ]
        )

        img = transform_img(img)
        original = transform_img(self.original)
        # 图片除以初始图片，并添加两通道的位置信息
        img = torch.div(img, original)
        position_x, position_y = torch.meshgrid(torch.arange(self.size_img[0]), torch.arange(self.size_img[1]),
                                                indexing='ij')
        # position = torch.stack((position_x, position_y))
        img = torch.cat((img, position_x.unsqueeze(0), position_y.unsqueeze(0)))

        angle_x = transform_label(angle_x)
        angle_y = transform_label(angle_y)

        return img, angle_x, angle_y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class GelSightGradientConvNet(nn.Module):

    def __init__(self, input_channels=5, out_channels=1):
        super(GelSightGradientConvNet, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.convNet = self._make_layers(4, 64)  # 论文中ConvNet x与 ConvNet y是分开训练的
        # self.convNet_y = self._make_layers(4, 64)

    def _make_layers(self, layers_num=4, channels=64):
        """
        依据原论文Section III.
        3x3 kernel used for 层1.1～1.4 2.1~2.4, 1.4层之后有一个kernel为2x2、stride=2的pooling层
        层3.1~3.4使用的是1x1的kernel，层3.3之后有一个50%概率的dropout层
        卷积层stride=1，文章中没有明确指出，使用默认步长
        文中默认全部使用64通道
        :param layers_num:
        :param channels:
        :return:
        """
        layers = [ConvLayer(self.input_channels, channels, kernel_size=3, stride=1, padding=1)]
        for _ in range(1, layers_num):
            layers.append(ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for _ in range(layers_num):
            layers.append(ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1))

        for _ in range(1, layers_num):
            layers.append(ConvLayer(channels, channels, kernel_size=1, stride=1, padding=0))

        layers += (nn.Dropout(p=0.5), ConvLayer(channels, self.out_channels, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*layers)

    def forward(self, img):
        angle = self.convNet(img)
        return angle


LR = 0.00001


def train(angle):
    # 论文中ConvNet x 与 ConvNet y是分开训练的, angle用来指示训练角度x还是角度y
    model = GelSightGradientConvNet().cuda()

    file_path = DATA_PATH

    train_data = BaselineDataset('train', file_path, (640, 480), (320, 240))
    val_data = BaselineDataset('val', file_path, (640, 480), (320, 240))
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True,  pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=16, shuffle=True,  pin_memory=True)
    trainval_loaders = {'train': train_loader, 'val': val_loader}
    trainval_sizes = {'train': train_data.__len__(), 'val': val_data.__len__()}

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=40, gamma=1.0 / 3.0)

    MSE = nn.MSELoss()
    # RMSE = cal_rmse
    # RMSE_LOG = cal_rmse_log
    # ABS_REL = cal_abs_rel
    # SQ_REL = cal_sq_rel
    # ACCURACY = cal_accuracies

    best_val_loss = sys.float_info.max
    train_loss_list, val_loss_list = [], []
    for epoch in range(EPOCH):
        print("Epoch %d" % epoch)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for step, data in enumerate(trainval_loaders[phase]):
                # print("in step %d" % step)
                inputs = data['img'].cuda()
                labels = data[angle].cuda()

                if phase == 'train':
                    predicts = model(inputs)
                else:
                    with torch.no_grad():
                        predicts = model(inputs)

                # loss = 0.5*MSE(predicts_x, labels_x) + 0.5*MSE(predicts_y, labels_y)
                loss = MSE(predicts, labels)
                # loss = L1(labels, predicts)
                running_loss += loss.item() * inputs.size(0)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / trainval_sizes[phase]

            if phase == 'train':
                train_loss_list.append(epoch_loss)
                scheduler.step()
            else:
                val_loss_list.append(epoch_loss)

            print("[{}] {} Epoch: {}/{} MSE Loss: {}".format(
                phase, angle, epoch + 1, EPOCH, epoch_loss))

            if phase == 'val':
                if best_val_loss > epoch_loss:
                    best_val_loss = epoch_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                    }, os.path.join(MODEL_PATH,
                                    '{}_best.pth.tar'.format(model.__class__.__name__ + angle)))
                    print("Save model at {}".format(
                        os.path.join(MODEL_PATH,
                                     '{}_best.pth.tar'.format(model.__class__.__name__ + angle))))
                    print('\n')


def convnet_test():
    angle_x_predictor = GelSightGradientConvNet().cuda()
    angle_y_predictor = GelSightGradientConvNet().cuda()

    file_path = DATA_PATH
    test_data = BaselineDataset('test', file_path, (640, 480), (320, 240))
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True, pin_memory=True)
    test_size = test_data.__len__()

    angle_x_predictor.load_state_dict(torch.load(MODEL_PATH + "/GelSightGradientConvNetangle_x_best.pth.tar")['state_dict'])
    angle_y_predictor.load_state_dict(torch.load(MODEL_PATH + "/GelSightGradientConvNetangle_y_best.pth.tar")['state_dict'])
    angle_x_predictor.eval()
    angle_y_predictor.eval()

    MSE = nn.MSELoss()
    test_loss = 0.0
    all_inputs = torch.tensor([]).cuda()
    all_labels = torch.tensor([]).cuda()
    all_outputs = torch.tensor([]).cuda()

    for step, data in enumerate(test_loader):
        inputs = data['img'].cuda()
        labels = data['depth'].cuda()

        with torch.no_grad():
            angle_x = angle_x_predictor(inputs)
            angle_y = angle_y_predictor(inputs)

        outputs = images_height_reconstruct(torch.tan(angle_x), torch.tan(angle_y), inputs)     # TODO
        loss = MSE(labels, outputs)
        test_loss += loss.item() * inputs.size(0)

        all_inputs = torch.cat((all_inputs, inputs), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)
        all_outputs = torch.cat((all_outputs, outputs), dim=0)

    test_loss /= test_size
    print("[TEST] MSE Loss: {} Test_Size: {}".format(test_loss, test_size))
    return all_inputs.cpu().numpy(), all_labels.cpu().numpy().squeeze(1), all_outputs.cpu().numpy().squeeze(1)


if __name__ == "__main__":
    model_list = ["06大球", "07中球", "08小球"]
    radius_map = {"06大球": 11.5, "07中球": 5.5, "08小球": 4}
    angle_maps_generator(model_list, DATA_PATH, radius_map)
    # generate_angle_maps(CONFIG_PATH, START_INDEX, DATA_PATH, IMAGE_SHAPE, RATIO, RADIUS)
    # img = torch.randn((8, 5, 640, 480))
    # model = GelSightGradientConvNet()
    # x, y = model(img)
    train('angle_x')
    train('angle_y')

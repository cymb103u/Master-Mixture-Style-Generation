
from __future__ import print_function

from torch.nn.functional import interpolate
from torchvision.transforms.functional import pad
from utils import get_config, pytorch03_to_pytorch04,get_all_data_loaders,\
                    slerp,lerp
from master_trainer import MUNIT_Trainer,MASTER_Trainer,MASTER_Trainer_v2
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and 0 for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MASTER', help="MUNIT|MASTER|MASTER_v2")
parser.add_argument('--gpu', type=int , default='1')
parser.add_argument('--interpolation', type=str ,help='slerp|lerp')
parser.add_argument('--one_img',type=str, default=None, help='the choice of one image. e.g., 1,2,17')
parser.add_argument('--multi_img',type=str, default=None, help='the choice of multiple image. e.g., 1,2,17')
opts = parser.parse_args()
cudnn.benchmark = True

GPU = opts.gpu
torch.cuda.set_device(GPU)
interpolation = slerp()if opts.interpolation =='slerp' else lerp()

# Load experiment setting
model_name = os.path.splitext(os.path.basename(opts.config))[0]
config = get_config(opts.config)
display_size = config['display_size']
if opts.one_img != None:
    one_img_str = opts.one_img.split(",")
    one_img = [int(i) for i in one_img_str]
    del one_img_str
    
if opts.multi_img != None:
    multi_img_str = opts.multi_img.split(",")
    multi_img = [int(i) for i in multi_img_str]
    del multi_img_str
# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
# elif opts.trainer == 'UNIT':
#     trainer = UNIT_Trainer(config)
elif opts.trainer == 'MASTER':
    trainer =MASTER_Trainer(config)
elif opts.trainer == 'MASTER_v2':
    trainer = MASTER_Trainer_v2(config)
else:
    sys.exit("Only support MASTER|MUNIT")
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

# weight load
try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen.load_state_dict(state_dict['gen'])
except:
    0
trainer.cuda()
trainer.eval()
encode = trainer.gen.encode
decode = trainer.gen.decode
# print(trainer)
# s_a_num = 0
# s_a = Variable(torch.randn(len(multi_img),8, 1, 1).cuda())
# s_b_num = 0
# s_b = Variable(torch.randn(len(one_img),8, 1, 1).cuda())
z_style_params = [0,0.2,0.4,0.6,0.8,1]
z_style = []
content_a_interp = []
content_b_interp = []
if opts.one_img == None and opts.multi_img == None:
    with torch.no_grad():
        for i, img_a in enumerate(test_loader_a):
            content_a, style_a = encode(img_a.cuda(), 1)
            for j, img_b in enumerate(test_loader_b):
                content_b, style_b = encode(img_b.cuda(), 2)
                img_b2a = decode(content_b,style_a,1)
                img_a2b = decode(content_a,style_b,2)
                all_img = torch.cat([img_a.cuda(),img_b.cuda(),img_b2a,img_a2b])
                # z_interp = None
                # for z in z_style_params:
                #     z_interp = interpolation(z,style_a,style_b)
                #     z_style.append(z_interp)
                #     if z==0 or z==1:
                #         content_a_interp.append(decode(content_a,z_interp,z+1))
                #         content_b_interp.append(decode(content_b,z_interp,z+1))
                #     else:
                #         content_a_interp.append(decode(content_a,z_interp,0))
                #         content_b_interp.append(decode(content_b,z_interp,0))
                # # print(z_style)
                # content_a_interp =torch.cat(content_a_interp)
                # content_b_interp =torch.cat(content_b_interp)
                
                # padding 代表間隔距離 
                vutils.save_image(img_b2a,f'outputs/{model_name}/generated/b2a/style_a{i:03d}_content_b{j:03d}.jpg',nrow=1,padding=0,normalize=True)
                vutils.save_image(img_a2b,f'outputs/{model_name}/generated/a2b/style_b{j:03d}_content_a{i:03d}.jpg',nrow=1,padding=0,normalize=True)
                # vutils.save_image(all_img,f'outputs/{model_name}/test/disentangle_a{i:03d}_b{j:03d}.jpg',nrow=1,padding=0,normalize=True)
                # vutils.save_image(content_a_interp,f'outputs/{model_name}/interpolation/content_a_interp_a{i:03d}_b{j:03d}.jpg',padding=0,normalize=True)
                # vutils.save_image(content_b_interp,f'outputs/{model_name}/interpolation/content_b_interp_a{i:03d}_b{j:03d}.jpg',padding=0,normalize=True)
                content_a_interp = []
                content_b_interp = []
elif opts.one_img != None and opts.multi_img != None:
    with torch.no_grad():
        for i, img_a in enumerate(test_loader_a):
            if i in one_img:
                s_a_num = 0
                s_b_num = 0
                content_a, style_a = encode(img_a.cuda(), 1)
                for j, img_b in enumerate(test_loader_b):
                    if j in multi_img:
                        content_b, style_b = encode(img_b.cuda(), 2)
                        # img_b2a = decode(content_b,style_a, 1)
                        # img_a2b = decode(content_a,style_b, 2)
                        # all_img = torch.cat([img_a.cuda(),img_b.cuda(),img_b2a,img_a2b])
                        # z_interp = None
                        # for z in z_style_params:
                        #     z_interp = interpolation(z,style_a,style_b)
                        #     z_style.append(z_interp)
                        #     if z==0 or z==1:
                        #         content_a_interp.append(decode(content_a,z_interp,z+1))
                        #         content_b_interp.append(decode(content_b,z_interp,z+1))
                        #     else:
                        #         content_a_interp.append(decode(content_a,z_interp,0))
                        #         content_b_interp.append(decode(content_b,z_interp,0))
                        # # print(z_style)
                        # content_a_interp =torch.cat(content_a_interp)
                        # content_b_interp =torch.cat(content_b_interp)
                        
                        # padding 代表間隔距離 
                        vutils.save_image(img_b2a,f'outputs/{model_name}/generated/b2a/style_a{i:03d}_content_b{j:03d}.jpg',nrow=1,padding=0,normalize=True)
                        vutils.save_image(img_a2b,f'outputs/{model_name}/generated/a2b/style_b{j:03d}_content_a{i:03d}.jpg',nrow=1,padding=0,normalize=True)
                        # vutils.save_image(all_img,f'outputs/{model_name}/test/disentangle_a{i:03d}_b{j:03d}.jpg',nrow=1,padding=0,normalize=True)
                        # vutils.save_image(content_a_interp,f'outputs/{model_name}/interpolation/content_a_interp_a{i:03d}_b{j:03d}.jpg',padding=0,normalize=True)
                        # vutils.save_image(content_b_interp,f'outputs/{model_name}/interpolation/content_b_interp_a{i:03d}_b{j:03d}.jpg',padding=0,normalize=True)
                        content_a_interp = []
                        content_b_interp = []

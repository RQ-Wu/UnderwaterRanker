from tkinter import E
import utils
import torch
import argparse
import os
import skimage.io as io
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--opt_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()

options = utils.get_option(args.opt_path)
options['model']['resume_ckpt_path'] = args.checkpoint_path
model = utils.build_model(options['model'])
filenames = os.listdir(args.input_path)

with open(args.save_path, 'w') as f:
    for filename in filenames:
        filepath = os.path.join(args.input_path, filename)
        img = Image.open(filepath)
        img_w, img_h = img.size[0], img.size[1]
            
        img = transforms.Resize((img_h//2, img_w//2))(img)
        img = transforms.ToTensor()(img).cuda().unsqueeze(0)
        inputs = utils.preprocessing(img)
        pred = model(**inputs)['final_result'][0][0][0]
        print(f'{filename}\t{pred}\n')
        f.write(f'{filename}\t{pred}\n')


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import argparse
import torch
import time
import glob
import cv2
import os

generator = torch.manual_seed(12345)

def process_input_image(img):
  width, height = img.size
  if width<224 or height<224:
    img = img.resize((width*2, height*2), Image.Resampling.LANCZOS)
  elif width>1024 or height>1024:
    img = img.resize((width//2, height//2), Image.Resampling.LANCZOS)
  return img


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input', required=True,
    type=str
  )
  parser.add_argument(
    '--prompt', required=True,
    type=str
  )
  parser.add_argument(
    '--sd_repo', required=False,
    default='/content/drive/MyDrive/ImageGen-Study-Avataar/checkpoints/sd-v1.5'
  )
  parser.add_argument(
    '--controlnet_repo', required=False,
    default=None
  )
  parser.add_argument(
    '--save_dir', required=False,
    type=str, default='results'
  )
  parser.add_argument(
    '--optimize', required=False,
    action='store_true', help='Set to true if needed to run inference with optimization techniques enabled. (Quantization, schedulers, etc.)'
  )
  parser.add_argument(
    '--inference_steps', required=False,
    type=int, default=50
  )
  args = parser.parse_args()

  # load checkpoints
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if args.controlnet_repo:
    controlnet = ControlNetModel.from_pretrained(
      args.controlnet_repo, torch_dtype=torch.float16, 
      local_files_only=True
    ).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
      args.sd_repo, controlnet=controlnet, torch_dtype=torch.float16,
      local_files_only=True,
      safety_checker = None,
      requires_safety_checker = False
    ).to(device)
  else:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
      args.sd_repo, torch_dtype=torch.float16,
      local_files_only=True,
      safety_checker = None,
      requires_safety_checker = False
    ).to(device)

  # read prompts
  prompts = []
  if os.path.isfile(args.prompt):
    with open(args.prompt) as f:
      prompts = [line.strip() for line in f.readlines()]
  else:
    prompts.append(args.prompt)
  
  # inference
  paths = []
  if os.path.isdir(args.input):
    paths = glob.glob(os.path.join(args.input, "*.png"))
  else:
    paths.append(args.input)
  # print(len(paths), len(prompts))
  assert len(paths)==len(prompts)
  os.makedirs(args.save_dir, exist_ok=True)
  for path, prompt in zip(paths, prompts):
    image = load_image(path)
    image = np.array(image)
    if len(image.shape) == 2:
      image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    if args.optimize:
      control_image = process_input_image(control_image)
      pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
      if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention()
    start = time.time()
    res = pipe(
      prompt, 
      num_inference_steps=args.inference_steps, 
      generator=generator, 
      image=control_image
    ).images[0]
    end = time.time()
    print('Inference time: ', end-start, 'seconds')
    if res.size != image.shape:
      res = res.resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS)
    save_path = os.path.join(args.save_dir, os.path.basename(path))
    res.save(save_path)

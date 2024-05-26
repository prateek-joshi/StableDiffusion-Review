# A Study on Image Generation using Stable Diffusion
A thorough analysis on the stable diffusion v1.5 image generation pipeline.
<b>This repo was setup and tested on Google Colab with the free T4 15GB GPU.</b><br/>
Report: [Google Doc](https://docs.google.com/document/d/1oJxdWYu5XIVtK3jcneoIMFBx5XNziv5Oha-CuC2WK-M/edit?usp=sharing)

## Setup
- Using versions:
    - `python=3.10.12`
    - `CUDA=12.2`
- `pip install -r requirements.txt`

## Checkpoints
Note: All checkpoints used are from huggingface.
- Stable Diffusion v1.5: `runwayml/stable-diffusion-v1-5`
- ControlNet
    - Canny edges: `lllyasviel/control_v11p_sd15_canny`
    - Normal map: `lllyasviel/control_v11p_sd15_normalbae`
    - Depth map: `lllyasviel/control_v11f1p_sd15_depth`

## Usage
1. <b>Creation of canny and normal maps from depth maps</b><br>
    ```
    python process.py \
    --img_source <path/to/folder/or/imagefile> \
    --canny \ # flag for running canny edge detection
    --normal # flag for running normal map generation
    ```
2. <b>Stable diffusion inference</b><br/>
    ```
    python inference.py \
    --input="<path/to/folder/or/imagefile>" \
    --prompt="luxury bedroom interior" # single text or text file with newline separated prompts \
    --save_dir="<path/to/save/folder>" \
    --optimize # Use optimization techniques to reduce compute and improve inference time \
    ```
3. <b>Guided stable diffusion inference using ControlNet</b><br/>
    ```
    python inference.py \
    --input="<path/to/folder/or/imagefile>" \
    --controlnet_repo=<controlnet/repo> \ # choose from the above specified controlnet checkpoints
    --prompt="prompt text" # single text or text file with newline separated prompts \
    --save_dir="<path/to/save/folder>" \
    --optimize # Use optimization techniques to reduce compute and improve inference time \
    ```
<b>Note: To use a different stable diffusion checkpoint, use the flag `--sd_repo` when running `inference.py`</b>

## Samples
<table>
    <tr>
        <th>Input Image</th>
        <th>Prompt</th>
        <th>Generated Image</th>
    </tr>
    <tr>
        <td><img src="results\depth\5.png" width=256 alt='depth map'></td>
        <td>"walls with cupboard"</td>
        <td><img src="results\inference\depth\5.png" width=256></td>
    </tr>
    <tr>
        <td><img src="results\canny\2.png" width=256 alt='canny edges'></td>
        <td>"luxury bedroom interior"</td>
        <td><img src="results\inference\canny\2.png" width=256></td>
    </tr>
    <tr>
        <td><img src="results\normal\6.png" width=256 alt='normal map'></td>
        <td>"room with chair"</td>
        <td><img src="results\inference\normal\6.png" width=256></td>
    </tr>
</table>
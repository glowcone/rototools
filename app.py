import os
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transparent_background import Remover

class Config:
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
    CHECKPOINTS_DIR = os.path.join(ASSETS_DIR, "checkpoints")
    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        "0.6b": "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        "1b": "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
    }

class ModelManager:

    @staticmethod
    def load_model(checkpoint_name: str):
        if checkpoint_name is None:
            return None
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        return F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)


bg_remover = Remover()
transform_fn = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[123.5 / 255, 116.5 / 255, 103.5 / 255], std=[58.5 / 255, 57.0 / 255, 57.5 / 255]),
])


def process_video(input_path, output_path, process_fn, progress=gr.Progress()):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'avc1'), fps, size)

    n = 0.0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break

        pil_frame = Image.fromarray(frame).convert('RGB')
        frame_output = process_fn(pil_frame)
        writer.write(np.array(frame_output))
        n += 1
        progress(n/cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    writer.release()
    return output_path


# gradio functions
def remove_bg(input_path, progress=gr.Progress()):
    output_path = 'temp-out-bg-removed.mp4'

    def process_frame(image: Image.Image):
        return bg_remover.process(image, type='[0, 0, 0]')

    process_video(input_path, output_path, process_frame, progress)

    return output_path

def get_normal_map(input_path, progress=gr.Progress()):
    output_path = 'temp-out-normal.mp4'

    def process_frame(image: Image.Image):
        seg_mask = np.array(bg_remover.process(image, type='map'))

        # Load models here instead of storing them as class attributes
        normal_model = ModelManager.load_model(Config.CHECKPOINTS['0.3b'])
        input_tensor = transform_fn(image).unsqueeze(0).to("cuda")

        # Run normal estimation
        normal_output = ModelManager.run_model(normal_model, input_tensor, image.height, image.width)
        normal_map = normal_output.squeeze().cpu().numpy().transpose(1, 2, 0)

        normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)
        normal_map_vis = ((normal_map_normalized + 1) / 2 * 255)

        seg_mask = seg_mask.astype(np.float64) / 255.0
        normal_map_vis = normal_map_vis.astype(np.float64)
        normal_map_vis *= seg_mask
        normal_map_vis = normal_map_vis.astype(np.uint8)

        return Image.fromarray(normal_map_vis)

    process_video(input_path, output_path, process_frame, progress)
    return output_path


# gradio ui
with gr.Blocks(theme='remilia/Ghostly') as demo:
    gr.Markdown("#### USC Ganek Lab")
    gr.Markdown("# Rototools")
    gr.Markdown("")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", loop=True, height=700)
        with gr.Column():
            with gr.Tab("Remove Background"):
                out_remove_bg = gr.Video(label="Result", interactive=False, loop=True, height=600)
                run_remove_bg_btn = gr.Button("Remove Background", variant='primary')
            with gr.Tab("Normal Map"):
                out_normal_map = gr.Video(label="Result", interactive=False, loop=True, height=600)
                run_normal_map_btn = gr.Button("Generate Normal Map", variant='primary')
            with gr.Tab("Depth Map"):
                out_depth_map = gr.Video(label="Result", interactive=False, loop=True, height=600)
                run_depth_map_btn = gr.Button("Generate Depth Map", variant='primary')

    run_remove_bg_btn.click(fn=remove_bg, inputs=input_video, outputs=out_remove_bg)
    run_normal_map_btn.click(fn=get_normal_map, inputs=input_video, outputs=out_normal_map)


if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    demo.launch(share=False)

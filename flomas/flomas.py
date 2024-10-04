import dataclasses
import gc
from typing import Tuple, List

import torch
from PIL import ImageDraw
from PIL.Image import Image
from torch.types import Device
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclasses.dataclass
class BboxResult:
    label: str
    bbox: Tuple[float, float, float, float]


def _release_resources():
    gc.collect()
    torch.cuda.empty_cache()


def draw_point(bbox_results: List[BboxResult], _image: Image):
    image = _image.copy()
    for result in bbox_results:
        x_min, y_min, x_max, y_max = result.bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        draw = ImageDraw.Draw(image)
        radius = 5
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill='red', outline='red'
        )
    return image


class Flomas:
    def __init__(
            self,
            checkpoint="microsoft/Florence-2-base-ft",
            revision='refs/pr/6',
            device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        if torch.cuda.is_available():
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.checkpoint = checkpoint
        self.revision = revision
        self.device = device

    def bboxes(self, prompt: str, image: Image):
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, trust_remote_code=True, revision=self.revision
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True, revision=self.revision
        )
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        text = f"<CAPTION_TO_PHRASE_GROUNDING> {prompt}"
        inputs = processor(text=text, images=image, return_tensors="pt").to(self.device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        _release_resources()

        labels = response['<CAPTION_TO_PHRASE_GROUNDING>']['labels']
        bboxes = response['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']

        return [
            BboxResult(
                label=label,
                bbox=bbox
            )
            for label, bbox in zip(labels, bboxes)
        ]

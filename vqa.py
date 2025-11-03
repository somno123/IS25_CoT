from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto"
).to(device)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
image = Image.open("cookie.png").convert("RGB") 

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image}, 
            {"type": "text", "text": "The challenge is to explain the picture to see if the person has dementia. Give important clues about important people, events, objects, etc. to express."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)

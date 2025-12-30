import os
import json
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set up device
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Moondream2 model and tokenizer
# model_name = "vikhyatk/moondream2"
# model = AutoModelForCausalLM.from_pretrained(model_name, revision="2025-04-14", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, revision="2025-04-14")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

 


# def generate_caption(image_path, length='normal'):
#     image = Image.open(image_path).convert("RGB")
#     # Use Moondream2's caption() method for detailed caption
#     caption_result = model.caption(image, length=length)
#     return caption_result["caption"]

# def generate_caption(image_path, length='normal'):
#     image = Image.open(image_path).convert("RGB")
#     with torch.no_grad():
#         caption_result = model.caption([image], tokenizer=tokenizer, length=length)
#     print(f"DEBUG: caption_result = {caption_result}")
#     # Try all reasonable options
#     if isinstance(caption_result, list):
#         if len(caption_result) > 0 and isinstance(caption_result[0], dict) and "caption" in caption_result[0]:
#             return caption_result[0]["caption"]
#         else:
#             return str(caption_result)
#     elif isinstance(caption_result, dict) and "caption" in caption_result:
#         return caption_result["caption"]
#     else:
#         return str(caption_result)



def generate_caption(image_path, length='normal'):
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        caption_result = model.caption([image], tokenizer=tokenizer, length=length)
    # Uncomment for debugging:
    # print(f"DEBUG: type={type(caption_result)}, value={caption_result}")
    # Handle: [{"caption": "..."}]
    if isinstance(caption_result, list):
        first = caption_result[0]
        if isinstance(first, dict) and "caption" in first:
            caption = first["caption"]
        elif isinstance(first, str):
            caption = first
        else:
            caption = str(first)
    elif isinstance(caption_result, dict) and "caption" in caption_result:
        caption = caption_result["caption"]
    elif isinstance(caption_result, str):
        caption = caption_result
    else:
        caption = str(caption_result)
    # If caption is a list, get first element
    if isinstance(caption, list) and len(caption) > 0:
        caption = caption[0]
    # If caption is a stringified list, extract the string
    if isinstance(caption, str) and caption.startswith("[") and caption.endswith("]"):
        try:
            import ast
            caption_list = ast.literal_eval(caption)
            if isinstance(caption_list, list) and len(caption_list) > 0:
                caption = caption_list[0]
        except Exception:
            pass
    return caption



# def generate_caption(image_path, length='normal'):
#     image = Image.open(image_path).convert("RGB")
#     with torch.no_grad():
#         #caption_result = model.caption(image, tokenizer=tokenizer, length=length)
#         caption_result = model.caption([image], tokenizer=tokenizer, length=length)
#     return caption_result[0]["caption"] #caption_result["caption"]


def create_dataset(image_folder, output_jsonl):
    records = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            caption = generate_caption(image_path)
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image_path": f"images/{filename}"},
                            {"type": "text", "text": "The line `"Describe the accident and fire cases in the image, give detailed and concise summary."` is a prompt or instruction provided to the user in the dataset creation process. This text is included in the dataset generation function `create_dataset` as part of the user role content. It instructs the user to describe any accident and fire cases they see in the image and provide a detailed and concise summary of what they observe. This prompt helps structure the data collection process by guiding users on what information to provide about the images they are describing.
                            Describe the accident and fire cases in the image, give detailed and concise summary."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": caption}
                        ]
                    }
                ]
            }
            records.append(record)
            print(f"Processed: {filename}")
    with open(output_jsonl, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    print(f"Saved dataset with {len(records)} samples to {output_jsonl}")
    return records

# Usage:
# Place your images in './images'
# create_dataset('./images', 'smolvlm_dataset.jsonl')
create_dataset("/home/hasan/drone_p4_implementation/images_sample/", 'clipvlm_dataset.jsonl')

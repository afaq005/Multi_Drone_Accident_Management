import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set this to the GPU you want to use

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
import torch

# --------- CONFIG ---------
model_id = "Salesforce/blip2-opt-2.7b"   # Small BLIP-2 model  gave nan values 

#model_id = "Salesforce/blip2-flan-t5-xl"  # works but gives prompt in inference
# Choose the BLIP-2 variant
 
train_csv = "train_fixed.csv"
valid_csv = "valid_fixed.csv"
image_folder = "/home/hasan/drone_p4_implementation/blip2_flikr/accident_dataset/images/"                # Folder with all images
output_dir = "./blip2_finetuned2"
max_length = 40 #40
batch_size = 1                           # 1 or 2 for 11GB GPU
num_epochs = 5



# --------- DATASET ---------
class CaptionDataset(Dataset):
    def __init__(self, csv_file, image_folder, processor):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.df.iloc[idx]['image'])
        caption = str(self.df.iloc[idx]['caption'])
        image = Image.open(image_path).convert("RGB")
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = item["input_ids"]
        return item

# --------- LOAD PROCESSOR & MODEL ---------
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    #torch_dtype=torch.float16,         # Use float16 for GPU memory efficiency
    device_map="auto" #"cuda:0"                  # Let HF/Accelerate place model on GPU
)

#Freeze ViT
for param in model.vision_model.parameters():
    param.requires_grad = False


# --------- DATASETS ---------
train_dataset = CaptionDataset(train_csv, image_folder, processor)
valid_dataset = CaptionDataset(valid_csv, image_folder, processor)


def blip2_data_collator(features):
    import torch
    batch = {}
    for k in features[0]:
        batch[k] = torch.stack([f[k] for f in features])
    #print("Batch labels unique:", torch.unique(batch["labels"]))
    return batch



# --------- TRAINING ARGS ---------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir=os.path.join(output_dir, "logs"),
    
    logging_strategy="steps",
    logging_steps=10,
    
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    save_total_limit=2,
    report_to="none",
)

 
#--------- TRAINER ---------
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset 
     
# )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=blip2_data_collator,
)


# --------- TRAIN ---------
trainer.train()

# --------- SAVE MODEL ---------
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

print("Fine-tuning complete. Model and processor saved to:", output_dir)

# train loop (working) that takes images from the root dirtectory, compatible with venv 3.11

# patched_trocr_finetune.py
import os, requests
from PIL import Image
import json
import numpy as np
import torch
from io import BytesIO
import time

from datasets import Dataset
import evaluate
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATASET_PATH = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset"
DATASET_PATH = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\right_side_up_workspace\dataset"
DATA_LABELS_PATH = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/d_labels.csv"
DATA_LABELS_PATH = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\right_side_up_workspace\d_labels.csv"
ALL_CARDS_PATH = r"C:\Users\Ben\Desktop\VSCodeCoding\MTGPriceReader\all_cards.json"

data = []
with open(ALL_CARDS_PATH, "r",encoding="utf-8") as file:
    reader = json.load(file)
    for line in reader:
        if not line.get("image_uris",0):continue
        url = line["image_uris"]
        if not url.get("png",0):continue
        url = url["png"]
        dp = {
            "image_path":url,
            "text": line["name"],
        }
        data.append(dp)

print(len(data))

hf_ds = Dataset.from_list(data)
split = hf_ds.train_test_split(test_size=0.1)
train_ds = split["train"]
val_ds = split["test"]

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

if getattr(processor, "tokenizer", None):
    if processor.tokenizer.cls_token_id is not None:
        model.config.decoder_start_token_id = int(processor.tokenizer.cls_token_id)
    if processor.tokenizer.pad_token_id is not None:
        model.config.pad_token_id = int(processor.tokenizer.pad_token_id)

for name, param in model.named_parameters():
    if name.startswith("encoder"):
        param.requires_grad = False

MAX_LABEL_LENGTH = 128 

def preprocess_fn(examples):
    images = []
    for p in examples["image_path"]:
        try:
            image_bytes = requests.get(p,stream=True).content
            b = BytesIO(image_bytes)
            image = Image.open(b).convert("RGB")
            crop_box = (
                int(0),
                int(0),
                int(3.82 * image.width / 5),
                int(image.height / 7),
            )
            image = image.crop(crop_box)
            images.append(image)
        except:
            print(f"Error with url {p}")
        time.sleep(7/100)

    proc_inputs = processor(images=images, return_tensors="np", padding=True)

    tokenized = processor.tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LABEL_LENGTH,
    )
    labels = np.array(tokenized["input_ids"], dtype=np.int64)

    pad_id = int(processor.tokenizer.pad_token_id)
    labels[labels == pad_id] = -100

    pixel_values = proc_inputs["pixel_values"].astype(np.float32).tolist()

    return {"pixel_values": pixel_values, "labels": labels.tolist()}

train_proc = train_ds.map(preprocess_fn, batched=True, remove_columns=["image_path", "text"])
val_proc = val_ds.map(preprocess_fn, batched=True, remove_columns=["image_path", "text"])

wer_metric = evaluate.load("wer")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(np.array(labels) == -100, int(processor.tokenizer.pad_token_id), np.array(labels))
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="my_trocr_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    eval_strategy="steps",           
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    generation_max_length=MAX_LABEL_LENGTH,   
    generation_num_beams=1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_proc,
    eval_dataset=val_proc,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)


try:
    sample_img_path = data[0]["image_path"]
    sample_img = Image.open(sample_img_path).convert("RGB")
    sample_inputs = processor(images=[sample_img], return_tensors="pt", padding=True)
    gen_kwargs = {"max_new_tokens": int(MAX_LABEL_LENGTH)}
    device = next(model.parameters()).device
    sample_pixel_values = sample_inputs["pixel_values"].to(device)
    model.eval()
    _ = model.generate(sample_pixel_values, **gen_kwargs)
    print("Dry-run generation OK. Proceeding to training.")
except Exception as e:
    print("Dry-run failed. Error:", e)
    raise

trainer.train()

trainer.save_model("finetuned")
processor.save_pretrained("finetuned")

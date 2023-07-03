

```shell
accelerate launch --mixed_precision="fp16" train_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="asset/lora" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=200 --checkpointing_steps=50 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-model-lora" \
  --validation_epochs=200 \
  --validation_prompt="A dog"
```
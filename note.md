
----------------------------------------------------------------
# Setup

```
python3 -m venv env

source env/bin/activate

pip install pendulum==2.1.2
pip install numpy==1.26.0
pip install matplotlib==3.8.0
pip install torchvision==0.15.2
pip install statistics==1.0.3.5
pip install tqdm==4.66.1
pip install monai==1.2.0
pip install pillow==10.0.1
pip install safetensors==0.3.3
pip install opencv-python==4.8.0.76
pip install pyyaml==6.0.1
pip install gradio==3.44.3
pip install torch==2.0.1

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
----------------------------------------------------------------

&nbsp;

&nbsp;





----------------------------------------------------------------
# Demo with legacy SAM

```
python3 -m tykim_scripts.01_demo_sam \
--image ./dataset/test/images/ring_test_1.jpg \
--box_xyxy 103 373 906 613
```

```
python3 -m tykim_scripts.01_demo_sam \
--image ./dataset/test/images/ring_test_2.jpg \
--box_xyxy 184 223 466 577
```



# Evaluation with legacy SAM


----------------------------------------------------------------





----------------------------------------------------------------
# LoRA Apply



# Train
applied config.yaml
```
DATASET:
  TRAIN_PATH: "./dataset/train"
  TEST_PATH: "./dataset/test"

SAM:
  CHECKPOINT: "./sam_vit_b_01ec64.pth"
  RANK: 512
TRAIN:
  BATCH_SIZE: 1
  NUM_EPOCHS: 200
```

```
python3 train.py
```



----------------------------------------------------------------





----------------------------------------------------------------
# Demo with LoRA SAM

```
python3 -m tykim_scripts.04_demo_lorasam \
--image ./dataset/test/images/ring_test_1.jpg \
--box_xyxy 103 373 906 613 \
--lora_weights lora_rank512.safetensors \
--rank 512
```

```
python3 -m tykim_scripts.04_demo_lorasam \
--image ./dataset/test/images/ring_test_2.jpg \
--box_xyxy 184 223 466 577 \
--lora_weights lora_rank512.safetensors \
--rank 512
```



# Evaluation


----------------------------------------------------------------
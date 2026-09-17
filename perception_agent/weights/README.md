# YOLOv11n Weights

Trained weights (`yolov11n_accident_fire.pt`) are **not committed to this
repository** due to size. Two options:

1. **Train from scratch** using `perception_agent/train_yolov11n.py` on the
   dataset produced by `dataset_finetuning/prepare_dataset.py`.
2. **Download the released checkpoint** (results in Table 1 of the paper,
   YOLOv11n: mAP@50 = 0.874 all classes) from the project's GitHub Releases
   page or Hugging Face Hub once published, and place it here as:

   ```
   perception_agent/weights/yolov11n_accident_fire.pt
   ```

The inference node (`inference_node.py`) and training script both read
`MDAM_YOLO_WEIGHTS` from `model/config.py`, which defaults to this path —
override it with an environment variable if you keep weights elsewhere:

```bash
export MDAM_YOLO_WEIGHTS=/path/to/your/weights.pt
```

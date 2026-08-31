# How to train and deploy model with rect is True

In non-square scenarios, enabling ```rect=True``` leads to more efficient computation, reduced latency, and enhanced feature learning—ultimately allowing higher accuracy without increasing model complexity or runtime.
In this tutorial, we will guide you through training and deploying a model with ```rect=True``` using esp-detection.


## Training

### Pretrain: Train model with rect=False(Optional)

```python

from train import Train

# pre-train your model
# take espdet_pico_160_288_cat for example
results = Train(
    dataset="cfg/datasets/coco_cat.yaml", # Path to dataset configuration file
    imgsz=288, # Image size for training
    epochs=900, # Number of training epochs
    rect=False,
    device="cpu", # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

```

### Train with rect=True

```python

from train import Train

# get the pre-trained weight
model_path = os.path.join(str(results.save_dir), "weights/best.pt") # optional, use the pre-trained model weights

# fine-tune your model
# take espdet_pico_160_288_cat for example
rect_results = Train(
    pretrained_path=model_path, # if you don't use pre-train, set pretrained_path=None
    dataset="cfg/datasets/coco_cat.yaml", # Path to dataset configuration file
    imgsz=[160, 288], # Image size for training, set to [h, w] when rect=True
    epochs=30, # Number of fine-tune epochs, 30~50
    rect=True,
    device="cpu", # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

```
## Export

```python

from deploy.export import Export

rect_model_path = os.path.join(str(rect_results.save_dir), "weights/best.pt")

# export the fine-tuned model
Export(
    model_path=rect_model_path,
    input_size=[160, 288], # [h, w]
)

```
## Quantization

```python

from deploy.quantize import quant_espdet
ONNX = rect_model_path.replace(".pt", ".onnx") # onnx path
quant_espdet(
    onnx_path=ONNX,
    target="esp32p4", # or "esp32s3"
    num_of_bits=8,
    device='cpu',
    batchsz=32,
    imgsz=[160, 288], # [h, w]
    calib_dir=calib_data,
    espdl_model_path="espdet_pico_160_288_cat.espdl", # .espdl model path
)
```

## Deployment

```python

from espdet_run import rename_project

esp_dl_url = "https://github.com/espressif/esp-dl.git"
esp_dl_path = "esp-dl"
# if not os.path.exists(esp_dl_path):
subprocess.run(["git", "clone", esp_dl_url, esp_dl_path])

examples_path = os.path.join(esp_dl_path, "examples")
models_path = os.path.join(esp_dl_path, "models")
custom_example_path = os.path.join(examples_path, class_name + "_detect")
custom_model_path = os.path.join(models_path, class_name + "_detect")
# create folder both in examples and models
os.makedirs(custom_example_path, exist_ok=True)
os.makedirs(custom_model_path, exist_ok=True)
# copy files from template to custom path
shutil.copytree("deploy/espdet_model_template", custom_model_path, dirs_exist_ok=True)
shutil.copytree("deploy/espdet_example_template", custom_example_path, dirs_exist_ok=True)

replacements = {
    "custom": class_name,
    "CUSTOM": class_name.upper(),
    "imgH": str(h), # 160
    "imgW": str(w), # 288
    "espdet.jpg": img,
    "espdet_jpg": os.path.splitext(img)[0] + "_jpg",
}

rename_project(Path(custom_example_path), replacements)
rename_project(Path(custom_model_path), replacements)

espdl_model_path = os.path.join(custom_model_path, "models/p4") if target == "esp32p4" else os.path.join(custom_model_path, "models/s3")
shutil.copy(espdl, espdl_model_path)

shutil.copy(img, os.path.join(custom_example_path, "main"))
```

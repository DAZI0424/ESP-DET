from ultralytics import YOLO
from nn.esp_tasks import custom_parse_model
import ultralytics.nn.tasks as tasks

def Train(pretrained_path=None, dataset="cfg/datasets/coco_cat.yaml", imgsz=224, **kwargs):
    """
    Train espdet_pico on customized dataset.
    :param pretrained_path: the path of pretrained .pt file, default is None.
    :return:
    """
    tasks.parse_model = custom_parse_model  # add ESP-customized block
    # load the model
    if pretrained_path not in [None, 'None']: # use pretrained weights
        model = YOLO(pretrained_path)
    else:
        model = YOLO('cfg/models/espdet_pico.yaml') # # build a new model from YAML if you don't need to load a pretrained model
    train_setting = dict( # you can set your own train settings here.
        data=dataset,
        epochs=1200, # set to a reasonable epoch
        imgsz=imgsz, # input img shape, 224 means input is 224*224. if you want to train with w ≠ h, you need to set rect=True and imgsz=[h, w]
        batch=128,
        device="cpu", # "1"
        optimizer='auto',
        close_mosaic=30,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.1,
        rect=False,
    )
    train_setting.update(kwargs)
    results = model.train(**train_setting)

    return results

if __name__ == '__main__':
    Train()


python espdet_run.py \
  --class_name mycat \
  --pretrained_path None \
  --dataset "cfg/datasets/coco_cat.yaml" \
  --size 224 224 \
  --target "esp32p4" \
  --calib_data "deploy/cat_calib" \
  --espdl "espdet_pico_224_224_mycat.espdl" \
  --img "espdet.jpg"
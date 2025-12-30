
# train
%cd /home/hasan/drone_p3_implementation/

!yolo task=detect mode=train model=yolo11n.pt data="/home/hasan/drone_p3_implementation/drone3_latest-2/data.yaml" epochs=200 imgsz=640 plots=True device=0 

# eval
%cd /home/hasan/drone_p3_implementation/

!yolo task=detect mode=val model="/home/hasan/drone_p3_implementation/runs/detect/train4/weights/best.pt" data="/home/hasan/drone_p3_implementation/drone3_latest-2/data1.yaml"  device=0

# eval
%cd /home/hasan/drone_p3_implementation/

!yolo task=detect mode=predict model="/home/hasan/drone_p3_implementation/runs/detect/Yolov11n_train/weights/best.pt" conf=0.5 source='/home/hasan/drone_p3_implementation/drone3_latest-2/test/images/' save=True device=0

# inference
!yolo task=detect mode=predict model="/home/hasan/drone_p2_implementation/runs/detect/main_trained_model/weights/best.pt" conf=0.5 source="/home/hasan/drone_p2_implementation/drone2-1/test/images/" save=True

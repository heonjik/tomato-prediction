import os
import pandas as pd
from ultralytics import YOLO
import cv2

# Training
def train_yolov8(data_path, model_path, trained_model_path, output_dir):
    model = YOLO(model_path)
    
    epochs = 10
    imgsz = 640
    batch = 16
    
    model.train(
    data=data_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    project=output_dir,
    name='tomato',
    cache=True
    )
    model.save(trained_model_path)
    print(f"Model trained and saved to {trained_model_path}")

def validate_yolov8(model_path, data_path):
    model = YOLO(model_path)
    imgsz=640
    metrics = model.val(data=data_path, imgsz=imgsz)
    print("Validation Results:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    print(f"List of mAP50-95 for each category: {metrics.box.maps}")

def test_yolov8(model_path, image_names, test_path, output_image_dir):
    model = YOLO(model_path)
    imgsz=640
    for image_name in image_names:
        source = os.path.join(test_path, image_name)
        output_image_path = os.path.join(output_image_dir, image_name)
        results = model.predict(source=source, imgsz=imgsz)

        result_img = results[0].plot()
        cv2.imwrite(output_image_path, result_img)
        print(f"Result image saved to {output_image_path}")


if __name__=='__main__':
    # Paths to your dataset and models
    cwd = os.getcwd()
    data_config = os.path.join(cwd, 'data.yaml')
    initial_model = os.path.join(cwd, 'yolov8n.pt')
    final_model = os.path.join(cwd, 'yolov8n_trained.pt')
    output_dir = os.path.join(cwd, 'runs', 'train')

    test_path = os.path.join(cwd, 'dataset', 'test', 'images')
    output_image_dir = os.path.join(cwd, 'test_result')
    image_names = ['riped_tomato_18.jpeg','riped_tomato_19.jpeg','riped_tomato_58.jpeg','riped_tomato_59.jpeg',
                   'riped_tomato_61.jpeg','riped_tomato_81.jpeg','riped_tomato_84.jpeg','riped_tomato_87.jpeg']

    # Train the model
    #train_yolov8(data_config, initial_model, final_model, output_dir)

    # Validate the model
    validate_yolov8(final_model, data_config)

    # Test the model on a new image or video
    #test_yolov8(final_model, image_names, test_path, output_image_dir)

# #coding:utf-8
# from ultralytics import YOLO
# import torch

# # 检查是否有可用的 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载预训练模型
# # 添加注意力机制，SEAtt_yolov8.yaml 默认使用的是n。
# # SEAtt_yolov8s.yaml，则使用的是s，模型。
# # model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml").load('/home/liumeilin/yolov8 ultralytics-main/yolov8n.pt').to(device)
# # model = YOLO("/home/liumeilin/yolov8 ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml").load('/home/liumeilin/yolov8 ultralytics-main/runs/detect/train246/weights/best.pt').to(device)
# model = YOLO("/home/liumeilin/yolov8 ultralytics-main/ultralytics/cfg/models/v8/yolov8 CLLAhead test.yaml").load('/home/liumeilin/yolov8 ultralytics-main/runs/detect/train246/weights/best.pt').to(device)
# # model = YOLO("/home/liumeilin/yolov8 ultralytics-main/ultralytics/cfg/models/v8/yolov8-dcn4CLLAhead.yaml").load('/home/liumeilin/yolov8 ultralytics-main/runs/detect/train215/weights/best.pt').to(device)
# # Use the model/home/liumeilin/yolov8 ultralytics-main/ultralytics/cfg/models/v8/yolov8-c2f-dcn4.yaml
# #/home/liumeilin/yolov8 ultralytics-main/ultralytics/cfg/models/v8/yolov8-dcn4CLLAhead.yaml
# if __name__ == '__main__':
#     # Use the model
#         # Use the model
#     #    results = model.train(data='/home/liumeilin/yolov8 ultralytics-main/my_data_alllast.yaml', epochs=300, batch=16) 
#     results = model.train(data='/home/liumeilin/yolov8 ultralytics-main/my_data_alllast.yaml', epochs=300, batch=16)  # 训练模型

#添加冻结层
from ultralytics import YOLO
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO('./weights/yolov8n.pt').load('yolov8n.pt').to(device)
model = YOLO('/home/liumeilin/yolov8 ultralytics-main/ultralytics/cfg/models/v8/yolov8 CLLAhead.yaml').load('/home/liumeilin/yolov8 ultralytics-main/runs/detect/train177/weights/best.pt').to(device)
#train257 没加bubble
# CLLAhead
# 未标注数据路径
# unlabeled_data = '/home/liumeilin/yolov8 ultralytics-main/datasets/fangtaoci/weibiaozhu'
# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_difffang.yaml'
# # 使用训练好的模型对未标注数据进行伪标注（生成预测标签）
# model.predict(source=unlabeled_data, save=True , conf=0.5,iou=0.6)
data = '/home/liumeilin/yolov8 ultralytics-main/my_data_alllast.yaml'
# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_yitihua_changaddbubble.yaml'
# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_diffyuanbing.yaml'
# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_alllast.yaml'
#  new data = '/home/liumeilin/yolov8 ultralytics-main/my_data_yitihua_chbupro.yaml'
# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_change.yaml'
# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_diffyuanzhu.yaml'

# data = '/home/liumeilin/yolov8 ultralytics-main/my_data_diffyzfen.yaml'
model.train(data = data, epochs=300, batch=8, save = True)
# def freeze_model(trainer):
#     # Retrieve the batch data
#     model = trainer.model
#     print('Befor Freeze')
#     for k, v in model.named_parameters():
#         print('\t', k,'\t', v.requires_grad)
        
        
#     freeze = 10
#     freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
#     for k, v in model.named_parameters():
#         v.requires_grad = True  # train all layers
#         if any(x in k for x in freeze):
#             print(f'freezing {k}')
#             v.requires_grad = False
#     print('After Freeze')
#     for k, v in model.named_parameters():
#         print('\t', k,'\t', v.requires_grad)
        

# model.add_callback("on_pretrain_routine_start", freeze_model)





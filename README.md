**已有数据集**

datsaets里包含了六种数据集，长方体、方形、圆柱、圆饼、全部、长方体迁移气泡数据集。

均进行了数据增强；同时包含了对应的yaml文件。

下载yolov8文件 安装requirements.txt


**改进-添加CLLAHead检测头**

1.ultralytics/nn/modules目录下放入CLLAhead.py文件
2.用本项目里的 task.py 替换 ultralytics/nn/tasks.py

**改进-添加WIOUV3损失函数**

loss wiou.py替换wiou.py

**训练**

根据需求选择不同的数据集 
在main.py中选择不同的数据集更改设置参数，并进行训练和验证.

**迁移学习**

数据集为：长方体迁移气泡数据集
transaddbu文件夹里包含了best.pt是在迁移学习过程中获得的最优模型.

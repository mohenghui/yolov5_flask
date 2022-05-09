# yolov5_flask
yolov5 v6.x在flask上的部署   
支持在线上传预测图片与视频功能以及摄像头检测功能      
上传后的视屏保存在/upload中   
结果保存在/inference/output文件夹中   
可自行替换自己训练的模型(pt)以及类别参数文件(我的是 my_person.yaml)，在camera.py中进行更改相应路径         
注意注意不要把项目放在中文路径下的文件夹   
CPU检测的用户需要在camera.py中修改select_device(0)为select_device('cpu')   
视频效果见bilibili   
https://www.bilibili.com/video/BV1eu411r7Qu?spm_id_from=333.999.0.0

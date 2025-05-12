# tijap-gpt
这只是个练习。 
This is just a practice.
---------------------------------
## 用于记录和梳理学习笔记
1. 看Andrej Karpathy大神的视频和其中提到的论文学习，源代码学习地址[https://github.com/karpathy/build-nanogpt]
2. 把每行代码搞明白，写自己理解的注释
3. 在Apple M4 Pro上试着运行，太慢了。。。只能放弃，去算力租赁平台上找合适的吧
4. 最终选择了智星云的GeForce RTX 4090 (24G)，系统镜像ubuntu，单机4卡
5. 调整参数，进行训练
6. 阅读学习DeepSeek-V3论文和源码[https://github.com/deepseek-ai/DeepSeek-V3]
7. 查看设备信息，GeForce RTX 4090支持FP8
8. 尝试DeepSeek 论文中提到的FP8 训练，相关版本如下：CUDA Version: 12.4, Driver Version: 550.127.05, torch 2.5.0, triton 3.1.0
9. 目前来看，内存占用减少了大约25%左右
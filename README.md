# GlobalPointer
参考了 [GlobalPointer_torch](https://github.com/xhw205/GlobalPointer_torch)，进一步实现：

1. 集成EffiGlobalPointer，GlobalPointer
2. Apex 混合训练
3. 对抗训练 fgm, awp等
4. EMA等平均移动权重策略

[中文医疗信息处理评测基准CBLUE](https://tianchi.aliyun.com/dataset/95414/hasRank)排行CBLUE2.0中CMeEE 线上f1为67.994，排名42。

<img src="https://aigonna.oss-cn-shenzhen.aliyuncs.com/blog/202306142016763.png" alt="image-20230614194925007" style="zoom:25%;" />

实验中，各个模型验证和测试集分数(有部分没记录)：

| 模型\训练方式                 | Valid F1 | Test F11（Online） |
| ----------------------------- | -------- | ------------------ |
| macbert + gp                  | 0.6556   | 67.073             |
| macbert + gp + fgm            | 0.6628   | 67.802             |
| macbert + gp + fgm + ema      | 0.6641   | 67.956             |
| cirbert + gp + fgm + ema      | 0.6498   | 67.787             |
| medbert-kd + gp + fgm + ema   | 0.6550   | 67.956             |
| macbert large+ gp + fgm + ema | 0.6599   | 67.994             |
| macbert egp+fgm+ema+sp        | 0.6584   | 67.630             |

从数据看来fgm、ema都能提升效果，同模型越大越好，macbert在Testset上表现最好，但是macbert + EffiGlobalPointer效果会差一点点。

1. 数据CMeEE经过`python data_utils.py`处理后就可以训练了。最好根据这个文件测试后再训练。

2. 训练：

```python
python train.py ----data_path ./data/CMeEE
```

3. 推理：

```python
python predict.py
```

代码还有些未解决问题，如没实现awp, EffiGlobalPointer使用Apex训练loss为NaN，有时间再修复！

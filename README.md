# Miniini DDSP Vocoders

<https://github.com/magenta/ddsp>

<https://github.com/YatingMusic/ddsp-singing-vocoders>

<https://github.com/yxlllc/pc-ddsp>

一个迷你ddsp vocoder，基于pc-ddsp的修改，推理时不需要ISTFT以及复杂的滤波器设计。<br>
这个项目的特征提取器支持谐波分离，运行

```bash
python harmonic_noise_extract.py --input_file /xxx 
```

即可进行谐波分离。

## 1. 安装依赖

首先下载pytroch，安装方法请参考 [official website](https://pytorch.org/)。<br>

然后运行：
```bash
pip install -r requirements.txt
```

## 2. Preprocessing

把训练数据放到如下目录: `data/train/audio`. 把测试数据放到如下目录: `data/val/audio`.<br>
使用VR模型进行谐波分离，把分离的谐波音频放到如下目录: `data/train/harmonic_audio ` 和 `data/val/harmonic_audio` <br>
(刚刚不是说支持谐波分离吗，为啥又要用到VR模型? 因为VR模型的谐波分离效果好于harmonic_noise_extract.py)。<br>

运行如下命令进行预处理:

```bash
python preprocess.py --config configs/SinStack.yaml
```

## 3. Training

```bash
python train.py --config configs/SinStack.yaml
```
如果爆显存了，可以把vocoder.py里的Sine_Generator换成Sine_Generator_Fast。

参数里有一个相位loss, 别开，学不会()
## 4. Inference

```bash
python main.py --model_path /xxx --input /audio --output /audio
```

# TPU-MobilenetSSD
## Environment
1. LattePanda Alpha (Ubuntu16.04) / RaspberryPi3 (Raspbian) / LaptopPC (Ubuntu16.04)
2. Edge TPU Accelerator (Supports multi-TPU)
3. USB Camera (Playstationeye)
## My articles
1.[I tested the operating speed of MobileNet-SSD v2 using Google Edge TPU Accelerator with RaspberryPi3 (USB2.0) and LaptopPC (USB3.1) (MS-COCO)](https://qiita.com/PINTO/items/dd6ba67643bdd3a0e595)  

2.[Structure visualization of Tensorflow Lite model files (.tflite)](https://qiita.com/PINTO/items/d74d92ece31c5f3fd040)  

3.[I wanted to speed up the operation of the Edge TPU Accelerator as little as possible, so I tried to generate a .tflite of MobileNetv2-SSDLite (Pascal VOC) and compile it into a TPU model. Part 1](https://qiita.com/PINTO/items/8368b0bc2d2d75a2f2e6)  

4.[Since I wanted to speed up the operation of the Edge TPU Accelerator as little as possible, I transferred and learned MobileNetv2-SSD / MobileNetv1-SSD + MS-COCO with Pascal VOC and generated .tflite. Docker Part 2](https://qiita.com/PINTO/items/8a91d79abe6e939ef01c)  

5.[Since we wanted to speed up the operation of the Edge TPU Accelerator as little as possible, I transferred and learned MS-COCO with Pascal VOC and generated .tflite, Google Colaboratory [GPU]. Part 3](https://qiita.com/PINTO/items/6eb6de95e3cda0e09c84)  

6.[Edge TPU Accelerator + custom model MobileNetv2-SSDLite .tflite generation 【Success】 Docker compilation Part.4](https://qiita.com/PINTO/items/1e34365cf46a3e660e25)  

7.[[150 FPS ++] Connect three Coral Edge TPU accelerators to infer parallelism and get ultra-fast object detection inference performance ーTo the extreme of useless high performanceー](https://qiita.com/PINTO/items/63b6f01eb22a5ab97901)  

## LattePanda Alpha Core m3 + USB 3.0 + Google Edge TPU Accelerator + MobileNet-SSD v2 + Async mode
**320x240**  
**about 80 - 90 FPS**  
**https://youtu.be/LERXuDXn0kY**  
  
![01](media/01.gif)
## LattePanda Alpha Core m3 + USB 3.0 + Google Edge TPU Accelerator + MobileNet-SSD v2 + Async mode
**640x480**  
**about 60 - 80 FPS**  
**https://youtu.be/OFEQHCQ5MsM**  
  
![02](media/02.gif)

## Core i7 + USB 3.0 + Google Edge TPU Accelerator / Multi-TPUs x3 + MobileNet-SSD v2 + Async mode
**320x240**  
**about 150 FPS++**  
**https://youtu.be/_qE9kmk8gUA**  
  
![03](media/03.gif)

# Environment construction procedure
```bash
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add-
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
$ sudo apt-get update
$ sudo apt-get upgrade edgetpu
$ wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz -O edgetpu_api.tar.gz --trust-server-names
$ tar xzf edgetpu_api.tar.gz
$ cd edgetpu_api
$ bash ./install.sh
```

# Usage
**MobileNet-SSD-TPU-async.py -> USB camera animation and inference are asynchronous (The frame is slightly off.)**  
**MobileNet-SSD-TPU-sync.py -> USB camera animation and inference are synchronous (The frame does not shift greatly.)**  

**If you use USB3.0 USBHub and connect multiple TPUs, it automatically detects multiple TPUs and processes inferences in parallel at high speed.**  
```bash
$ git clone https://github.com/PINTO0309/TPU-MobilenetSSD.git
$ cd TPU-MobilenetSSD
$ python3 MobileNet-SSD-TPU-async.py
```

# Reference
- Get started with the USB Accelerator https://coral.withgoogle.com/tutorials/accelerator
- Models https://coral.withgoogle.com/models/
- Edge TPU Model Compiler https://coral.withgoogle.com/web-compiler/
- API demos https://coral.withgoogle.com/tutorials/edgetpu-api/#api-demos
- Edge TPU Benchmark https://coral.withgoogle.com/tutorials/edgetpu-faq/

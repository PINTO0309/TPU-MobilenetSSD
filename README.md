# TPU-MobilenetSSD
## Environment
1. LattePanda Alpha (Ubuntu16.04) / RaspberryPi3 (Raspbian) / LaptopPC (Ubuntu16.04)
2. Edge TPU Accelerator
3. USB Camera (Playstationeye)
## My articles
1.[I tested the operating speed of MobileNet-SSD v2 using Google Edge TPU Accelerator with RaspberryPi3 (USB2.0) and LaptopPC (USB3.1) (MS-COCO)](https://qiita.com/PINTO/items/dd6ba67643bdd3a0e595)  

2.[Structure visualization of Tensorflow Lite model files (.tflite)](https://qiita.com/PINTO/items/d74d92ece31c5f3fd040)  

3.[I wanted to speed up the operation of the Edge TPU Accelerator as little as possible, so I tried to generate a .tflite of MobileNetv2-SSDLite (Pascal VOC) and compile it into a TPU model. Part 1](https://qiita.com/PINTO/items/8368b0bc2d2d75a2f2e6)  

4.[Since I wanted to speed up the operation of the Edge TPU Accelerator as little as possible, I transferred and learned MobileNetv2-SSD / MobileNetv1-SSD + MS-COCO with Pascal VOC and generated .tflite. Part 2](https://qiita.com/PINTO/items/8a91d79abe6e939ef01c)  

5.[Since we wanted to speed up the operation of the Edge TPU Accelerator as little as possible, I transferred and learned MS-COCO with Pascal VOC and generated .tflite, Google Colaboratery [GPU]. Part 3](https://qiita.com/PINTO/items/6eb6de95e3cda0e09c84)  

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

# Environment construction procedure
```bash
$ wget http://storage.googleapis.com/cloud-iot-edge-pretrained-models/edgetpu_api.tar.gz
$ tar xzf edgetpu_api.tar.gz
$ cd python-tflite-source
$ bash ./install.sh
```

# Usage
**MobileNet-SSD-TPU-async.py -> USB camera animation and inference are asynchronous (The frame is slightly off.)**  
**MobileNet-SSD-TPU-sync.py -> USB camera animation and inference are synchronous (The frame does not shift greatly.)**  
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

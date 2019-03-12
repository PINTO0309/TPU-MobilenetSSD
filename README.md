# TPU-MobilenetSSD
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
```bash
$ git clone https://github.com/PINTO0309/TPU-MobilenetSSD.git
$ cd TPU-MobilenetSSD
$ python3 MobileNet-SSD-TPU-async.py
```

# MISOnet
Unofficial Pytorch Multi-microphone complex spectral mapping for utterance-wise and continuous speech separation(MISO-BF-MISO)
https://arxiv.org/abs/2010.01703

## Todo
- [x] MISO1 implementation (seperation Network)
- [x] Speaker Alignment System
- [x] MVDR implementation
- [x] MISO3 implementatino (enhancement Network)
- [ ] Speaker counting Network
- [x] SMS-WSJ Dataset generation
- [ ] LibriCSS Dataset generation

## Requirements
 - Python>=3.8.0
 - Pytorch>=1.10.0
 - (optional) virtualenv
 
## Training

0. (Optional) Setup Virtualenv
```
sudo pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
```

1. Setup python packages environments
```
pip install -r requirements.txt
```

2. Run (todo)
```
python run.py --config=./config
```

3. Spectrogram # Example of 3_441c040w_445c040o_0.wav amoung test_eval92 (sms_wsj)
- Obervation
<img src="https://user-images.githubusercontent.com/67786803/149370478-77ed9d46-76cb-4ff4-baf5-4146ecbef723.jpg" width="300" height="300">

- Clean Source 1 & 2
<span>
<img src="https://user-images.githubusercontent.com/67786803/149371375-9371212c-9a78-4424-b58f-169d6c2b9ce3.jpg" width="300" height="300">
<img src="https://user-images.githubusercontent.com/67786803/149371411-e2c0e646-d4cd-47b9-b4d5-740af458e0ab.jpg" width="300" height="300">
</span>

<audio>
    <source src='https://user-images.githubusercontent.com/67786803/149378199-85120003-e907-47d5-904d-84342c649454.mp4'>
</audio>

- MISO1 Model Output Source 1 & 2
<span>
<img src="https://user-images.githubusercontent.com/67786803/149370815-0eef8473-4933-4c25-ab36-a348e9fe8e9f.jpg" width="300" height="300">
<img src="https://user-images.githubusercontent.com/67786803/149370856-ec84f2d4-7df2-4f5b-bd65-e32f14887f95.jpg" width="300" height="300">
</span>

- MVDR Beamformer Output Source 1 & 2
<span>
<img src="https://user-images.githubusercontent.com/67786803/149370919-8d0a17a0-aecc-4d95-8430-227ab4568fb4.jpg" width="300" height="300">
<img src="https://user-images.githubusercontent.com/67786803/149370962-831f5eff-3806-46a9-a63f-cf11003f9604.jpg" width="300" height="300">
</span>

- MISO3 Model Output Source 1 & 2
<span>
<img src="https://user-images.githubusercontent.com/67786803/149371113-110b1915-b9e1-4f4f-872c-04d1c3176091.jpg" width="300" height="300">
<img src="https://user-images.githubusercontent.com/67786803/149371589-8f5b660e-f19f-4d44-8d84-01b3683b101d.jpg" width="300" height="300">
</span>

## Reference
https://github.com/kaituoxu/Conv-TasNet
https://github.com/fgnt/sms_wsj
https://github.com/chenzhuo1011/libri_css

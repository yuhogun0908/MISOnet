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

3. Spectrogram
  - Obervation
  ![mix](https://user-images.githubusercontent.com/67786803/142854365-fd342767-c4cb-4222-9f52-0ee3dd57ba57.jpg){: width="300" height="300"}

  - Clean Source 1
  ![clean1](https://user-images.githubusercontent.com/67786803/142854420-c8e5ea9c-8016-48b4-a952-421078054d08.jpg){: width="300" height="300"}

  - Clean Source 2
  ![clean2](https://user-images.githubusercontent.com/67786803/142854443-d979702e-7182-4373-a01a-5f37da2d9dd7.jpg){: width="300" height="300"}
 
  - MISO1 Model Output Source 1 
  ![MISO1](https://user-images.githubusercontent.com/67786803/142854505-debc5819-2475-41b5-9f90-f6b52b08e355.jpg){: width="300" height="300"}

  - MISO1 Model Output Source 2
  ![MISO2](https://user-images.githubusercontent.com/67786803/142854547-7443024d-e43f-47c9-97f9-442c1a82b0ad.jpg){: width="300" height="300"}

  - MVDR Beamformer Output Source 1
  ![beamout1](https://user-images.githubusercontent.com/67786803/142854587-7efb9afa-bc9f-42fc-94f5-fb7b61c3fded.jpg){: width="300" height="300"}

  - MVDR Beamformer Output Source 2
  ![beamout2](https://user-images.githubusercontent.com/67786803/142854613-9657a16c-f602-4ed2-86d6-f5afa52e4c8d.jpg){: width="300" height="300"}



## Reference
https://github.com/kaituoxu/Conv-TasNet
https://github.com/fgnt/sms_wsj
https://github.com/chenzhuo1011/libri_css

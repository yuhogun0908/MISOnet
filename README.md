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

## Reference
https://github.com/kaituoxu/Conv-TasNet
https://github.com/fgnt/sms_wsj
https://github.com/chenzhuo1011/libri_css

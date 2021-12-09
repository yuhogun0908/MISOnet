python run.py \
-c ./config \
-d SMS_WSJ \
-m Extraction \
-u 1 \
-n ./runs/12_08_MISO3_Batch20_MISO1_1115_MISO1_targetScaled_overlap3_4_stftScalerevise \
-t MISO1 # MISO1 Beamforming, MISO2, MISO3

# -d REVERB_2MIX  #RIR_mixing #SMS_WSJ
# -m Extraction # Train # Test
# -u 1 0 Whether use gpu
# -n tensorboard
# -t Select Model to train or test 


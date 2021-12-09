python run.py \
-c ./config \
-d SMS_WSJ \
<<<<<<< HEAD
-m Extraction \
-u 1 \
-n ./runs/12_08_MISO3_Batch20_MISO1_1115_MISO1_targetScaled_overlap3_4_stftScalerevise \
-t MISO1 # MISO1 Beamforming, MISO2, MISO3

# -d REVERB_2MIX  #RIR_mixing #SMS_WSJ
# -m Extraction # Train # Test
# -u 1 0 Whether use gpu
# -n tensorboard
# -t Select Model to train or test 

=======
-m Train \
-u 1 \
-n ./runs/Test \
-t MISO3 # MISO1 Beamforming, MISO2, MISO3

#REVERB_2MIX  #RIR_mixing #SMS_WSJ
#Test_MISO_1  # Extraction # Train # Test
# Whether use gpu 
>>>>>>> 7431d9618a519d5bf78594445b6810a1a197388d

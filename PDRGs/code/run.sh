#!/bin/bash


cd ~/PDNetworks/PDRGs

FOLDER=Human_PPI
DEVICE=0
P_EDGEDROP=0.9
PER_BS=0.8



for ((SEED=0; SEED<=9; SEED+=1)); do
  python ./code/main.py \
    --rand_seed $SEED \
    --device $DEVICE \
    --pretrained 0 \
    --preprocess svd \
    --dirnet      ./data/ppi_350K.txt \
    --dirresult   ./output/$FOLDER/$SEED/true_power_concat_p_edgedrop${P_EDGEDROP}_bs${PER_BS}/ \
    --dirlog      ./output/$FOLDER/$SEED/true_power_concat_p_edgedrop${P_EDGEDROP}_bs${PER_BS}/ \
    --dirproteinfamily ./data/ppi_uniprot_protein_family.pkl \
    --dirfuncgeno ./data/qtl.conf      \
    --n_comp 3800 \
    --p_edgedrop $P_EDGEDROP \
    --p_bs $PER_BS \
    --patience 15 \
    --n_nei 2 \
    --lr 1e-4 \
    --hidden_size 2048 1024 \
    --K 1200 \
    --epochs 20000 \
    --use_weight False
done

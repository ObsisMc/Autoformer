#!/bin/bash

# M
python -u pred_autoformer.py --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_96_96 --model Autoformer --data custom --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des 'Exp' --idx 0 --pred_idx 168

# S
python -u pred_autoformer.py --root_path ../../dataset --features S --data_path ECL.csv --model_id ECL_168_168 --model Autoformer --data custom --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --pred_idx 168 --target MT_320
python -u pred_autoformer.py --root_path ../../dataset --features S --data_path WTH.csv --model_id WTH_168_168 --model Autoformer --data custom --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --pred_idx 168 --target WetBulbCelsius
python -u pred_autoformer.py --root_path ../../dataset/ETT-small --features S --data_path ETTh1.csv --model_id ETTh1_168_168 --model Autoformer --data ETTh1 --seq_len 168 --label_len 168 --pred_len 168 --e_layers 2 --d_layers 1 --factor 3 --enc_in 1 --dec_in 1 --c_out 1 --des 'Exp' --pred_idx 168 --target OT

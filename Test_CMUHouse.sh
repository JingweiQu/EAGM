cd graph_generator && CUDA_VISIBLE_DEVICES=1 python CMUHouse_EF.py
cd .. && CUDA_VISIBLE_DEVICES=1 python EAGM_Test.py --dataset CMUHouse
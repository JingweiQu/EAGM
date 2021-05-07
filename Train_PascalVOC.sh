cd graph_generator && CUDA_VISIBLE_DEVICES=2 python PascalVOC_EF.py --split train
cd .. && CUDA_VISIBLE_DEVICES=2 python EAGM_Train.py --dataset PascalVOC
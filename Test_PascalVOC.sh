cd graph_generator && CUDA_VISIBLE_DEVICES=2 python PascalVOC_EF.py --split test
cd .. && CUDA_VISIBLE_DEVICES=2 python EAGM_Test.py --dataset PascalVOC
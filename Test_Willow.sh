cd graph_generator && CUDA_VISIBLE_DEVICES=0 python Willow_EF.py
cd .. && CUDA_VISIBLE_DEVICES=0 python EAGM_Test.py --dataset Willow
from datasets import Crack500Dataset

dataset = Crack500Dataset(data_root='/dataset/siyuanchen/research/data/crack/Crack500')
print(dataset.load_data_list())
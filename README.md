# EMGCN
Code of the paper: ***Multi-order Graph Convolutional Networks for Knowledge Graph Alignment***.

# Environment

* python>=3.5 
* networkx == 1.11 (**important!**) 
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Running

```
python -u network_alignment.py --dataset_name zh_en --source_dataset data/networkx/zh_enDI/zh/graphsage/ --target_dataset data/networkx/zh_enDI/en/graphsage --groundtruth data/networkx/zh_enDI/dictionaries/groundtruth EMGCN --sparse --log 
```

# Dataset
You can download our processed dataset from: https://drive.google.com/file/d/12XL08tB8zplCNhzLE-9qbsFFum7RoV6r/view?usp=sharing. 

# Citation

Please politely cite our work as follows:

*Nguyen Thanh Tam, Huynh Thanh Trung, Hongzhi Yin, Tong Van Vinh, Darnbi Sakong, Bolong Zheng, Nguyen Quoc Viet Hung. Multi-order Graph Convolutional Networks for Knowledge Graph . In: TKDE 2020.*

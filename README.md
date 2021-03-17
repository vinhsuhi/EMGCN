# EMGCN
Code of the paper: ***Entity Alignment for Knowledge Graphs with Multi-order Convolutional Networks***.

# Environment

* python>=3.5 
* networkx == 1.11 (**important!**) 
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Dataset
You can download our processed dataset from: https://drive.google.com/file/d/12XL08tB8zplCNhzLE-9qbsFFum7RoV6r/view?usp=sharing. 

# Running

```
python -u network_alignment.py --dataset_name zh_en --source_dataset data/networkx/zh_enDI/zh/graphsage/ --target_dataset data/networkx/zh_enDI/en/graphsage --groundtruth data/networkx/zh_enDI/dictionaries/groundtruth EMGCN --sparse --log 
```

# Citation

Please politely cite our work as follows:

*Nguyen Thanh Tam, Huynh Thanh Trung, Hongzhi Yin, Tong Van Vinh, Darnbi Sakong, Bolong Zheng, Nguyen Quoc Viet Hung. Entity Alignment for Knowledge Graphs with Multi-order Convolutional Networks . In: TKDE 2020.*

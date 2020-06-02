# EMGCN
Code of the paper: ***Entity Alignment for Knowledge Graphs with Multi-order Convolutional Network***.

# Environment

* python>=3.5 
* networkx == 1.11 (**important!**) 
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Running

```
python -u network_alignment.py --dataset_name zh_en --source_dataset data/networkx/zh_enID/zh/graphsage/ --target_dataset data/networkx/zh_enID/en/graphsage --groundtruth data/networkx/zh_enID/dictionaries/groundtruth EMGCN --sparse --log 
```

# Dataset
You can download our processed dataset from: https://drive.google.com/file/d/12XL08tB8zplCNhzLE-9qbsFFum7RoV6r/view?usp=sharing. 

# Citation

Please politely cite our work as follows:

*Nguyen Thanh Tam, Huynh Thanh Trung, Hongzhi Yin, Tong Van Vinh, Darnbi Sakong, Bolong Zheng, Nguyen Quoc Viet Hung. Entity Alignment for Knowledge Graphs with Multi-order Convolutional Network. In: TKDE 2020.*

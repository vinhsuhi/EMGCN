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

@article{nguyen2020entity,
  title={Entity alignment for knowledge graphs with multi-order convolutional networks},
  author={Nguyen, Tam Thanh and Huynh, Thanh Trung and Yin, Hongzhi and Van Tong, Vinh and Sakong, Darnbi and Zheng, Bolong and Nguyen, Quoc Viet Hung},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2020},
  publisher={IEEE}
}

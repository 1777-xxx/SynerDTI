#SynerDTI：A Synergistic Deep Learning Framework for Drug-Target Interaction Prediction via Global Feature Coordinated Attention Mechanism
## Framework
![10 20绘图](https://github.com/user-attachments/assets/e0073550-7347-4fbf-a9bc-9714ab132f84)
## System Requirements
```
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```

## Using
$ python main.py --cfg "configs/Syner.yaml" --data ${dataset} --split ${split_task}

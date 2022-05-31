# Contrastive sensor transformer for predictive maintenance of industrial assets
This repository is the official implementation of Contrastive Sensor Transformer (CST) published at ICASSP 2022. 

<details><summary>Abstract (click to expand)</summary>
<p>

  CST is a novel approach for learning useful representations for robust fault identification without using task-specific labels. We explore sensor transformations for pre-training in a self-supervised contrastive manner, where the similarity between the original signal instance and its augmented version is maximized. We demonstrate that the powerful transformer architecture applied to condition monitoring data learns highly useful embedding that perform exceptionally well for fault detection in low labeled data regimes and for the identification of novel fault types. Our approach obtains an average of 75\% accuracy on the considered bearing benchmark datasets while using less than 2\% of the labeled instances.

</p>
</details>

![header image](https://github.com/Zaharah/Contrastive-Sensor-Transformer/blob/main/cst.JPG)

## Dataset Preparation
To prepare the datasets, please follow code given in jupyter notebook above.

## Reference
If you use this repository, please consider citing:

<pre>@INPROCEEDINGS{9746728,
  author={Bukhsh, Zaharah},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Contrastive Sensor Transformer for Predictive Maintenance of Industrial Assets}, 
  year={2022},
  volume={},
  number={},
  pages={3558-3562},
  doi={10.1109/ICASSP43922.2022.9746728}}</pre>





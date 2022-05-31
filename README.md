# Contrastive sensor transformer for predictive maintenance of industrial assets

This repository is the official implementation of Contrastive Sensor Transformer (CST) published at ICASSP 2022.  CST is a novel approach for learning useful representations for robust fault identification without using task-specific labels. We explore sensor transformations for pre-training in a self-supervised contrastive manner, where the similarity between the original signal instance and its augmented version is maximized. We demonstrate that the powerful transformer architecture applied to condition monitoring data learns highly useful embedding that perform exceptionally well for fault detection in low labeled data regimes and for the identification of novel fault types. Our approach obtains an average of 75\% accuracy on the considered bearing benchmark datasets while using less than 2\% of the labeled instances.



<img src=method.pdf width=80%/>


# ACLearning

The official PyTorch implementation of our **Interspeech** 2026 paper:

**Cross Domain Few-shot Class Incremental Audio Classification via Adversarial Contrastive Learning** [[paper]]

## Abstract
Existing methods of Few-shot Class-incremental Audio Classification (FCAC) assume that samples of both base and incremental classes are in the same domain. Namely, their distributions are assumed to be the same. However, in practical cases, there is often a domain-shift between these two types of samples. In this paper, we explore the problem of Cross Domain FCAC (CD-FCAC) in which samples of base and incremental classes have domain-shift. Furthermore, we propose an adversarial contrastive training strategy for CD-FCAC, which enables the model to have better generalization on unseen domains and classes. The proposed model is composed of an embedding extractor and an expandable classifier. The embedding extractor is trained in base session but frozen in incremental sessions. The expandable classifier is trained in both base and incremental sessions. Experiments are conducted on six pairs of datasets. Results show that our method exceeds nine baseline methods in average accuracy. 



## Datasets


Three audio datasets, including FSC-89, NSynth-100 and LS-100, are adopted as experimental datasets to evaluate the performance of different methods, which have been widely used in previous works for audio classification. To facilitate reimplementation of the results of this paper, the details of these three audio datasets are described at three websites . What's more, they can be downloaded from the above three websites and be freely used for research purpose. 

https://www.modelscope.cn/datasets/pp199124903/LS-100/summary 

https://www.modelscope.cn/datasets/pp199124903/FSC-89/summary 

https://www.modelscope.cn/datasets/pp199124903/NSynth-100/summary 

  

## Code

- run 
    ```bash
    python train.py --config=.configs/l2n.yml
    ```


## Contact
Yanxiong Li (eeyxli@scut.edu.cn) and Yongjie Si (eeyongjiesi@mail.scut.edu.cn)
School of Electronic and Information Engineering, South China University of Technology, Guangzhou, China


### **Project's name: Convexity Regularizer for Neural Optimal Transport**

### **Member of Project**
- Nikita Gushchin-TA

Team's name: BinKoff (Group 31)
- [Bintang Alam Semesta W.A.M](https://www.linkedin.com/in/bintangalamsemestawisranam/))-1st year master student, Advanced Computational Science
- [Fidele Koffivi Gbagbe](https://www.linkedin.com/in/koffivi)-1st year master student, Advanced Computational Science
 
### **Research Idea**
The method proposed in the research paper uses adversarial training (kind of GANs). which is not very stable to compute the optimal transport map $T$. From the theory, the method's optimal "discriminator" must be convex, and its gradient can be used for inverse mapping from the target distribution to the source distribution.

### **Objectives**
Add a convexity regularizer in the loss of neural optimal transport algorithm to improve its stability during direct, i.e. source-to-target or training-to-test map as well as to test the quality of generated images during inverse, i.e. target-to-source or test-to-training map.

### **Environment (Dependencies)**
- [POT](https://pythonot.github.io/)
- [wandb](https://wandb.ai/site)
- [torch](https://pytorch.org/docs/stable/torch.html)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [numpy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [matplotlib](https://matplotlib.org/)

how to install:
- `pip3 install -r dependencies.txt`
or
- `pip install -r dependencies.txt`

### **Repository Structure**
- NOT-Group 31
  - notebooks & custom modules ---documentation of source codes and corresponding modules
    - jupyter notebook files:
      - [`NOT_training_strong.ipynb`](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/notebooks%20%26%20custom%20modules/NOT_training_strong.ipynb)
      - [`NOT_inverse_map_true.ipynb`](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/notebooks%20%26%20custom%20modules/NOT_inverse_map_true.ipynb)
      - [`NOT_Regularized.ipynb`](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/notebooks%20%26%20custom%20modules/NOT_Regularized.ipynb)
      - [`color_MNIST.ipynb`](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/notebooks%20%26%20custom%20modules/color_MNIST.ipynb)
    - modules:
      - src
      - stats
    
  - images
    - fixed-test-images-53900
    - random-test-images-53900
    - digits-3
    - digits-3 to digits-2
    - [Reg-NOT algorithm](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/images/Reg-NOT%20algorithm.PNG)
    - [NOT algorithm](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/images/NOT%20algorithm.PNG)
 
  - references
  - LICENSE
  - README.md
 
  - [presentation](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/Presentation-Group%2031-Convexity%20Regularizer%20NOT.pdf)
  - [project report](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/Project%20Report-Group%2031-Convexity%20Regularizer%20NOT.pdf)

### **Dataset**
[color-MNIST](https://github.com/bin-koff/convexity-regularizer-NOT/blob/main/notebooks%20%26%20custom%20modules/color_MNIST.ipynb)

### **Credits**
- [weight and biases](https://wandb.ai/) --> data logging for machine learning 
- [FID score](https://arxiv.org/abs/1706.08500) --> a metric for evaluating the quality of generated images and specifically developed to evaluate the performance of generative adversarial networks

### **References**
1. [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Daniil Selikhanovych](https://scholar.google.com/citations?user=ZpZhN3QAAAAJ&hl=en), [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru). [Neural Optimal Transport](https://arxiv.org/pdf/2201.12220.pdf).Eleventh International Conference on Learning Representations.arXiv:2201.12220v3 [cs.LG] 1 Mar 2023 

2. Alexander Korotin. [Neural Optimal Transport Presentation](https://www.youtube.com/watch?v=tMfn_Tbcakc&ab_channel=ATRC) (August, 9th 2022)

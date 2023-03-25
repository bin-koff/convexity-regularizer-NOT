### **Project's name: Convexity Regularizer for Neural Optimal Transport**

### **Member of Project**
- Nikita Gushchin-TA

Team's name: BinKoff (Group 31)
- [Bintang Alam Semesta W.A.M](https://www.linkedin.com/in/bintang-alam-semesta-w-a-m-62b20a145)-1st year master student, ACS
- [Fidele Koffivi Gbagbe](https://www.linkedin.com/in/koffivi)-1st year master student, ACS
 
### **Research Idea**
The method proposed in the research paper uses adversarial training (kind of GANs). which is not very stable to compute the optimal transport map $T$. From the theory, the method's optimal "discriminator" must be convex, and its gradient can be used for inverse mapping from the target distribution to the source distribution.

### **Objectives**
Add a convexity regularizer in the loss of neural optimal transport algorithm to improve its stability during transport training and the high-quality of the inverse target-to-input mapping.

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

### **Folder Structure**
- images---documentation for snippet figures (format: `.jpeg` or `.png`)
   - ddd
   - ddd
   - dd
   
- models---documentation of the outputs (format:) 
  - ddd
  - ddd
  - ddd

- notebooks---documentation of source codes (format: `.ipynb`)
  - ddd
  - ddd
  - ddd

- resources---documentation of scientific papers from journals/conferences
  - dddd
  - dddd
  - ddd

- LICENSE
- README.md

### **Dataset**

### **Running an Experiment**

- Full Experiment:
  - pre-processing
  - training
  - evaluating (testing)
  - saving the model


### **Credits**
(give credit here,........)

### **References**
1. [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Daniil Selikhanovych](https://scholar.google.com/citations?user=ZpZhN3QAAAAJ&hl=en), [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru). [Neural Optimal Transport](https://arxiv.org/pdf/2201.12220.pdf).Eleventh International Conference on Learning Representations.arXiv:2201.12220v3 [cs.LG] 1 Mar 2023 \\
2. Alexander Korotin. [Neural Optimal Transport Presentation](https://www.youtube.com/watch?v=tMfn_Tbcakc&ab_channel=ATRC) (August, 9th 2022)

### **Project's name: Convexity Regularizer for Neural Optimal Transport**

### **Member of Project**
- Nikita Guschin : TA

Team's name: BinKoff (Group 31)
- Bintang Alam Semesta W.A.M: 1st year master student, ACS
- Fidele Koffivi GBagbe : 1st year master student, ACS
 
### **Research Idea**
The method proposed in the research paper uses adversarial training (kind of GANs). which is not very stable to compute the optimal transport map $T$. From the theory, the method's optimal "discriminator" must be convex, and its gradient can be used for inverse mapping from the target distribution to the source distribution.

### **Objectives*
Add a convexity regularizer in the loss of neural optimal transport algorithm to improve its stability during transport training and the high-quality of the inverse target-to-input mapping.

### **Environment (Dependencies)**
`POT==0.8.2`
`wandb`
`torch==1.13.1+cu116`
`numpy>=1.21.6`
`tqdm>=4.64.1`
`matplotlib>=3.2.2`

how to install:
`pip3 install -r dependencies.txt`
or
`pip install -r dependencies.txt`

### **Downloaded Data**

### **Training a Model**

### **Evaluating a Model**

### **Running a Full Experiment**
Full Experiment (pre-processing, training, save, and testing a model)

### **Experiment with Custom Datasets**

### **TO-DO Activities**

### **Acknowledgements**
(give credit here,..........)

### **References**

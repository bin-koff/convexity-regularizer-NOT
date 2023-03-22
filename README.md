# **Convexity Regularizer for Neural Optimal Transport** 

### **Background**
Using NN (Neural Networks) to tackle continuous optimal transport problem is a promising approach especially for unpaired style-transfer problem. This method learns a one-to-one mapping between the source and target data distributions but uses adversarial  training (similar to GANs), which is not very stable. Unlike GANs, this method's optimal "discriminator" must be convex, and its gradient can be used for inverse mapping from the target distribution to the source distribution. To address the stability issue, it's necessary to insert a convexity regularizer (kind of gradient penalty in WGAN-GP) in the loss of neural optimal transport to improve its staiblity during optimal transport training while maintaining the-quality of the inverse target-to-input mapping.

keywords: one-to-one map, inverse target-to-input-mapping, adversarial training, convex optimal transport, convexity regularizer, WGAN-GP

### **Environment**

### **Downloaded Data**

### **Training a Model**

## **Evaluating a Model**

## **Running a Full Experiment**
Full Experiment (pre-processing, training, save, and testing a model)

## **Experiment with Custom Datasets**

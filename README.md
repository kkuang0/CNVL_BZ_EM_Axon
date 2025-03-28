# Boston University Computational Neuroscience and Vison Lab, Human Systems Neuroscience Lab
# Machine learning approaches to systematically study short- and long-range cortical pathways and identify axon pathology in autism

Axon pathology is at the core of disruptions in cortical connectivity in autism spectrum disorder (ASD). However, the extent and distribution of disruption in a) short- and long-range cortical pathways and b) pathways linking association cortices or other cortices and subcortical structures are unknown. Neuroanatomical analysis of high-resolution features of individual axons, such as density, trajectory, branching patterns, and myelin in multiple cortical pathways, are labor-intensive and time-consuming. This limits large-scale studies that otherwise can help identify core pathways that are altered in ASD and likely mechanisms that underly neural communication disruption. To automate and optimize analysis and visualization of patterns of disruption, we customized machine learning techniques to quantify the requisite power of multiscale optical and electron microscopy for accurately classifying neurotypical and ASD postmortem brain histopathology sections.

To recreate the environment, run:
```
conda env create -f environment.yml
conda activate axon_EM
```

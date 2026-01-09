# Oxford Flowers 102 Classification 🌸

Fine-grained image classification on the Oxford Flowers 102 dataset using PyTorch and transfer learning.

## Dataset
- 8189 images across 102 flower categories
- Moderate class imbalance (40–258 images per class)
- Source:
  - Visual Geometry Group, University of Oxford: [Official download page](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)  
  - Publication: [Nilsback & Zisserman, 2008](https://www.robots.ox.ac.uk/~men/papers/nilsback_icvgip08.pdf)

Images loaded from local directory structure with provided 'imagelabels.mat' file.

## Key Results
### Pretrained ResNet-50 + full fine-tuning (best performing)
- Top-1 Accuracy: **97.71%** sdt 0.51% (mean over 10 stratified runs)
- Top-5 Accuracy: **99.69%** sdt 0.13%
- Macro F1-score: 97.46%
- Model size: ~91 MB (~23.7M parameters)
- Inference time: ~0.25 ms/image

### Custom LightResCNN (lightweight model trained from scratch)
- Top-1 Accuracy: **90.07%** sdt 1.22% (mean over 10 stratified runs)
- Top-5 Accuracy: **97.71%** sdt 0.39%
- Macro F1-score: 88.90%
- Model size: ~21 MB (~5.4M parameters)
- Inference time: ~0.09 ms/image (fastest)

**Full analysis available in the notebook**:  
training and validation curves, confusion matrices, ROC curves (macro/micro), detailed error analysis on misclassified classes, comprehensive commentary, interpretations and conclusions after each experiment.

## Technologies
- PyTorch, Torchvision
- NumPy, SciPy
- Matplotlib, Seaborn
- Scikit-learn

## Notebook Structure
The entire project is implemented in a single, well-organized Jupyter Notebook for clarity and reproducibility:
- Clear sectioning using Markdown headers
- Reusable functions defined at the top of each major section
- Modular design within the notebook - easy to extract into .py files if needed
- Clean visualizations and interpreted results after each step
- Automatic GPU support and mixed precision training

## Notebook
Complete pipeline with EDA, training, evaluation and model comparisons:  
[flowers_classification.ipynb](flowers_classification.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mateusz-zarebski/flowers102-classification/blob/main/flowers_classification.ipynb)

## How to Run
pip install -r requirements.txt

jupyter notebook flowers_classification.ipynb

⭐ Star if you like it!  
Author: Mateusz Zarębski [main profile](https://github.com/mateusz-zarebski)

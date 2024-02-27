# GIST-Python

## Getting started

### Installation
1. [Download and Install Python](https://www.python.org/downloads/)
2. Clone the repository
    ```
   git clone https://github.com/simran-ss-sandhu/GIST-Python
   ```
3. Navigate to the folder
    ```
   cd GIST-Python
   ```
4. Install Python packages and dependencies
    ```
   pip install -r requirements.txt
   ```

### Usage
GIST descriptors can be extracted by using the `GIST.py` file locally with:
```python 
from GIST import calculate_gist_descriptors

# paths of images to calculate GIST descriptors from
img_paths = ['img1.png', 'C:\\Users\\username\\Pictures\\img2.jpg']

descriptors = calculate_gist_descriptors(img_paths)

# descriptors[i] = the descriptor of the image at img_paths[i]
img1_descriptor = descriptors[0]
img2_descriptor = descriptors[1]
```

## Acknowledgements
- [A.Oliva and A.Torralba, Modelling the shape of the scene: a holistic representation of the spatial envelope, 2001](http://people.csail.mit.edu/torralba/code/spatialenvelope/)
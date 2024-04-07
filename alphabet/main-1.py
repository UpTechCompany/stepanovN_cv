import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops 
from collections import defaultdict
from pathlib import Path

def has_line(region, horizontal=True):
    return 1. in np.mean(region.image, int(horizontal))

def holes_count(region):
    inv = np.logical_not(region.image)
    labeled = label(inv)
    holes = np.max(labeled)
    return holes

def extractor(region):
    area = region.area / region.image.size
    euler = (region.euler_number+ 1)/2
    eccentrcity = region.eccentricity
    if eccentrcity < 0.4:
        eccentrcity = 0
    peremeter = region.perimeter / region.image.size
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    line_v = has_line(region, False)
    holes = holes_count(region)
    if eccentrcity == 0:
        holes = 0
    return np.array([area, euler, eccentrcity, peremeter, cy, cx, line_v, holes])

def distance(p1, p2):
    return ((p1-p2) ** 2).sum() ** 0.5

def classificator(prop, classes):
    klass = None
    min_d = 10**16
    for cls in classes:
        d = distance(prop, classes[cls])
        if d < min_d:
            klass = cls
            min_d = d
    return klass

image = plt.imread('alphabet_small.png').mean(2)
image[image==1] = 0
image[image!=0] = 1
labeled = label(image)
regions = regionprops(labeled)
                                       # Набор символов из правильного кода
classes = {'A': extractor(regions[2]), # 21
           'B': extractor(regions[3]), # 25
           '8': extractor(regions[0]), # 23
           '0': extractor(regions[1]), # 10
           '1': extractor(regions[4]), # 31
           'W': extractor(regions[5]), # 12
           'X': extractor(regions[6]), # 15
           '*': extractor(regions[7]), # 22
           '-': extractor(regions[9]), # 20
           '/': extractor(regions[8])} # 21
alphabet = plt.imread('alphabet.png').mean(2)
alphabet[alphabet>0] = 1
labeled = label(alphabet)
regions = regionprops(labeled)

result = defaultdict(lambda: 0)
path = Path('.') / 'result-1'
path.mkdir(exist_ok=True)
plt.figure()
for i, region in enumerate(regionprops(labeled)):
    symbol = classificator(extractor(region), classes)
    result[symbol] += 1
    plt.clf()
    plt.title(f'{symbol=}')
    plt.imshow(region.image)
    plt.tight_layout()
    plt.savefig(path/f'{i}.png')

correct = {'A':21, 'B':25, '8':23, '0':10, '1':31, 'W':12, 'X':15, '*':22, '-':20, '/':21}
if correct == result:
    print('Полное совпадение')
print(result)
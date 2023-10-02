import tensorflow_datasets as tfds
import os
import random
import skimage.io as io
import numpy as np
import json

_CATEGORIES_FILE = 'C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Tareas/Tarea 2/sketchy_classes.json'

# Change file names and add extras
_TRAIN_FILE_PHOTO = 'C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Tareas/Tarea 2/train_photo.txt'
_TRAIN_FILE_SKETCH = 'C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Tareas/Tarea 2/train_sketch.txt'
_TEST_FILE_PHOTO = 'C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Tareas/Tarea 2/test_photo.txt'
_TEST_FILE_SKETCH = 'C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Tareas/Tarea 2/test_sketch.txt'



class tfds_sketchy(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def get_categories(self):
        with open(_CATEGORIES_FILE) as json_file:
            data = json.load(json_file)
        categories = data.keys()
        return categories

    def _info(self):
        categories = self.get_categories()
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'sketch': tfds.features.Image(shape=(256, 256, 3)),
                'photo': tfds.features.Image(shape=(256, 256, 3)),
                'label': tfds.features.ClassLabel(names=categories),
            }),
            supervised_keys=('sketch', 'photo', 'label'),
            )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(_TRAIN_FILE_PHOTO,_TRAIN_FILE_SKETCH),
            'test': self._generate_examples(_TEST_FILE_PHOTO, _TEST_FILE_SKETCH),
            }
    
    def _generate_examples(self, fname_photo, fnmae_sketch):
        with open(fname_photo) as flist_photo, open(fnmae_sketch) as flist_sketch:
            clase_vieja = None
            i = 0
            for imcode1 in flist_photo:
                if (i + 1)  % 10000 == 0 :
                    print('{} {}'.format(fname_photo, i+1))
                clase = imcode1.strip().split('\t')[-1]
                if(clase != clase_vieja):
                    flist_sketch.seek(0)
                    lista_de_misma_clase = []
                    for imcode2 in flist_sketch:
                        if(imcode2.strip().split('\t')[-1] == clase):
                            lista_de_misma_clase.append(imcode2)
                clase_vieja = clase
                i+=1
                choose_sketch = random.choice(lista_de_misma_clase)
                yield i, {
                    'photo': io.imread(imcode1.strip().split('\t')[0]),
                    'sketch': io.imread(choose_sketch.strip().split('\t')[0]),
                    # 'negative': io.imread(negative_sketch.strip().split('\t')[0]),
                    'label': int(clase.strip()),
                }
        # print(photo, sketch)
        # print()
        # with open(fname) as flist :
        #     for i , f in enumerate(flist):       
        #         if (i + 1)  % 10000 == 0 :
        #             print('{} {}'.format(fname, i+1))                                      
        #         data = f.strip().split('\t')
        #         sketch_path = data[0].strip()
        #         sketch = io.imread(sketch_path)
        #         photo_path = data[1].strip()
        #         photo = io.imread(photo_path)
        #         label = int(data[2].strip())

        #         yield i, {
        #             'sketch': sketch,
        #             'photo': photo,
        #             'label': label,
        #         }

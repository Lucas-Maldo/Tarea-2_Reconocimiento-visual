import os
import random
import json

PHOTO_VER = 'tx_000100000000'
SKETCH_VER = 'tx_000100000000'
RANDOM_SEED = 1234
DIRECCION_PHOTO = "C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Datasets/256x256/photo"
DIRECCION_SKETCH = "C:/Lucas things/University/Computacion/Reconocimiento visual deep learning/Datasets/256x256/sketch"

if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    photo_folder = os.path.join(DIRECCION_PHOTO, PHOTO_VER)
    sketch_folder = os.path.join(DIRECCION_SKETCH, SKETCH_VER)
    # Obtenemos el listado de clases del dataset
    classes = os.listdir(photo_folder)
    # Creamos un diccionario para codificar las clases con números, class -> int
    class_dict = {}
    for i, name in enumerate(classes):
        class_dict[name] = i
    # Guardamos el diccionario en caso de necesitar decodificar las clases
    with open('./sketchy _classes.json', 'w') as f:
        json.dump(class_dict, f)
    # Seleccionamos las clases con las que vamos a entrenar
    random.shuffle(classes)

    ## Dataset train y test para photo y sketch
    train_photo_file = open('./train_photo.txt', 'w')
    test_photo_file = open('./test_photo.txt', 'w')
    train_sketch_file = open('./train_sketch.txt', 'w')
    test_sketch_file = open('./test_sketch.txt', 'w')
    for current_class in classes:
        # Separamos un 80% de las fotos para train y un 20% para validation
        photo_dir = os.path.join(photo_folder, current_class)
        photo_list = os.listdir(photo_dir)
        random.shuffle(photo_list)
        cutoff = int(len(photo_list)*0.8)
        train_photos = photo_list[:cutoff]
        test_photos = photo_list[cutoff:]
        sketch_dir = os.path.join(sketch_folder, current_class)
        sketch_list = os.listdir(sketch_dir)
        random.shuffle(sketch_list)
        cutoff = int(len(sketch_list)*0.8)
        train_sketches = sketch_list[:cutoff]
        test_sketches = sketch_list[cutoff:]
        for photo in train_photos:
            # Si coinciden con una foto de train, guardamos en train
            photo_abs = os.path.abspath(os.path.join(photo_dir, photo))
            train_photo_file.write(f'{photo_abs}\t{class_dict[current_class]}\n')
        for photo in test_photos:
            # Si coinciden con una foto de train, guardamos en train
            photo_abs = os.path.abspath(os.path.join(photo_dir, photo))
            test_photo_file.write(f'{photo_abs}\t{class_dict[current_class]}\n')
        for sketch in train_sketches:
            # Si coinciden con una foto de train, guardamos en train
            sketch_abs = os.path.abspath(os.path.join(sketch_dir, sketch))
            train_sketch_file.write(f'{sketch_abs}\t{class_dict[current_class]}\n')
        for sketch in test_sketches:
            # Si coinciden con una foto de train, guardamos en train
            sketch_abs = os.path.abspath(os.path.join(sketch_dir, sketch))
            test_sketch_file.write(f'{sketch_abs}\t{class_dict[current_class]}\n')

    train_photo_file.close()
    test_photo_file.close()
    train_sketch_file.close()
    test_sketch_file.close()



import os
import tensorflow as tf
from tensorflow.python.ops.gen_functional_ops import RemoteCall
import tensorflow_datasets as tfds
import configparser
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------------
def map_func(example_serialized):    
#     features_map=tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(None, None, 3)),
#                                               'label': tfds.features.ClassLabel(names=range(100))})
#     features = tf.io.parse_example(example_serialized, features_map)
    image_anchor = example_serialized['photo']
    image_positive = example_serialized['sketch']    
    image_anchor = tf.image.resize_with_pad(image_anchor, 256, 256)
    image_positive = tf.image.resize_with_pad(image_positive, 256, 256)
    image_anchor = tf.image.random_crop(image_anchor, size = [224, 224, 3])
    image_positive = tf.image.random_crop(image_positive, size = [224, 224, 3])
    image_positive = tf.cast(image_positive, tf.float32)    
    image_anchor = tf.cast(image_anchor, tf.float32)
    return image_anchor, image_positive 

        
AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
# import drive.MyDrive.models.siamese as siamese
import drive.MyDrive.Computacion.Deep_learning.Tarea_2.siamese as siamese
# import drive.MyDrive.siamese as siamese

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, choices = ['RESNET'], required = True)
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    model_name = args.model
    config_model = config[model_name]
    config_data = config['DATA']
    dataset_name = config_data.get('DATASET')
    
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu
        
    # ds = tfds.load('tfds_sketchy', data_dir = "/content/drive/MyDrive/tensorflow_datasets")    
    ds = tfds.load('tfds_sketchy', data_dir = "/content/drive/MyDrive/Computacion/Deep_learning/Tarea_2/tensorflow_datasets")  
    ds_train = ds['train']
    ds_test = ds['test']    
    ds_train = (
        ds_train.shuffle(1024)
        .map(map_func, num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO) )
    
    # #----------------------------------------------------------------------------------
    # model_dir =  config_model.get('MODEL_DIR')    
    # if not config_model.get('EXP_CODE') is None :
    #     model_dir = os.path.join(model_dir, config_model.get('EXP_CODE'))
    # model_dir = os.path.join(model_dir, dataset_name, model_name)
    # if not os.path.exists(model_dir) :
    #     os.makedirs(os.path.join(model_dir, 'ckp'))
    #     os.makedirs(os.path.join(model_dir, 'model'))
    #     print('--- {} was created'.format(os.path.dirname(model_dir)))
    # #----------------------------------------------------------------------------------
    
    if gpu_id >= 0 :
        print("Using GPU")
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            model = siamese.Siamese(config_model, config_data)
            model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9))    
            #Nuevo                        
            model.build()
            history = model.fit(ds_train, epochs = config_model.getint('EPOCHS'))


            encoder = model.get_test_encoder()

            sketch_embeddings = []
            photo_embeddings = []
            sketch_labels = []
            photo_labels = []

            i = 1
            print("Aqui llega: ", len(ds_test))

            for example in tfds.as_numpy(ds_test):  # Adjust based on actual use case
                print("Iteracion: ", i)

                photo_embeddings.append(encoder.predict(np.expand_dims(example['photo'], axis=0))[0])
                sketch_embeddings.append(encoder.predict(np.expand_dims(example['sketch'], axis=0))[0])

                photo_labels.append(example['label'])
                sketch_labels.append(example['label'])
                i +=1

            sketch_embeddings = np.array(sketch_embeddings)
            photo_embeddings = np.array(photo_embeddings)
            sketch_labels = np.array(sketch_labels)
            photo_labels = np.array(photo_labels)

            similitudes_totales = []
            clasificaciones_totales = []
            print(tfds.as_numpy(ds_test))
            # Aqui calcula las diferencias entre los embeddings de cada sketch con los embeddings de todas las photos
            for idx, query_embedding in enumerate(sketch_embeddings):

                similitudes = np.linalg.norm(photo_embeddings - query_embedding, axis=1)
                clasificacion = (photo_labels == sketch_labels[idx]).astype(int)

                similitudes_totales.append(-similitudes)
                clasificaciones_totales.append(clasificacion)

            sketchy_classes = list(json.load(open("sketchy_classes.json", "r")).keys())

            def calculate_average_precision(predictions, ground_truth):
              sorted_indices = np.argsort(predictions)[::-1]
              sorted_ground_truth = ground_truth[sorted_indices]
              
              precisions = []
              recalls = []
              verdadero = 0 
              falso = 0
              
              for i, label in enumerate(sorted_ground_truth):
                  if label == 1:
                      verdadero += 1
                  else:
                      falso += 1
                  precision = verdadero / (verdadero + falso)
                  recall = verdadero / np.sum(ground_truth)
                  precisions.append(precision)
                  recalls.append(recall)
              
              precisions = np.array(precisions)
              recalls = np.array(recalls)
              average_precision = np.sum(precisions * np.gradient(recalls))

              return average_precision

            ap_por_imagen = []
            for i, j in zip(similitudes_totales, clasificaciones_totales):
                ap_por_imagen.append(calculate_average_precision(i, j))


            AP_per_class = {}
            for i, ap in enumerate(ap_por_imagen):
                class_name = sketchy_classes[sketch_labels[i]]
                if(class_name not in AP_per_class.keys()):
                    AP_per_class[class_name] = [ap, 1]
                else:
                    AP_per_class[class_name] = [AP_per_class[class_name][0]+ap, AP_per_class[class_name][1]+1]
            
            for i in AP_per_class.keys():
                print("clase: ", i, ", AP por clase: ", AP_per_class[i][0]/AP_per_class[i][1])

            mean_average_precision = np.mean(ap_por_imagen)
            print("Mean average precision: ", mean_average_precision)

        def calculate_recall_at_1(queries, items, query_labels, item_labels):
            similitudes = []
            class_recall_at_1 = {}
            
            for query, query_label in zip(queries, query_labels):

                similitudes = np.linalg.norm(items - query, axis=1)
                sorted_similitudes = np.argsort(similitudes)
                # lowest_distances = similitudes[sorted_similitudes[0]]
                lowest_distances_index = sorted_similitudes[0]

                recall_at_1 = (query_label == item_labels[lowest_distances_index]).astype(int)
                number_of_total_relevant_photos = np.count_nonzero(item_labels == query_label)
                

                if(query_label not in class_recall_at_1.keys()):
                    class_recall_at_1[query_label] = [recall_at_1, number_of_total_relevant_photos]
                else:
                    class_recall_at_1[query_label] = [class_recall_at_1[query_label][0]+recall_at_1, class_recall_at_1[query_label][1]+number_of_total_relevant_photos]

            return class_recall_at_1


        # Calcular Recall@1
        total_recal_at_1 = calculate_recall_at_1(sketch_embeddings, photo_embeddings, sketch_labels, photo_labels)
        recall_at_1_total = 0
        for k, v in total_recal_at_1.items():
            recall_at_1_total += v[0]/v[1]
            print("recall@1 class: ",sketchy_classes[k],"puntaje: ", v[0]/v[1])

        print("Total recall@1: ", recall_at_1_total/len(list(total_recal_at_1.values())) )


        #Calcular recall-precision graph
        def compute_recall_precision_curve(indexes, s_labels, p_labels, n):

            recall_precision_curve = []

            for n in range(1, 10):
                recalls = []
                precisions = []

                for indice_de_photos, sketch_label in zip(indexes, s_labels):
                    # Check if any of the top-N predicted images are in the set of true relevant images
                    correct_retrievals = sum(1 for p in indice_de_photos[:n] if p_labels[p] == sketch_label)
                    precision = correct_retrievals / n
                    recall = correct_retrievals / sum(1 for p in p_labels if p == sketch_label)

                    recalls.append(recall)
                    precisions.append(precision)

                # Calculate average Recall and Precision at this N
                avg_recall = sum(recalls) / len(recalls)
                avg_precision = sum(precisions) / len(precisions)

                recall_precision_curve.append((avg_recall, avg_precision))

            return recall_precision_curve

        top_10_predictions_per_sketch_index = []
        for idx, query_embedding in enumerate(sketch_embeddings):
            similares = np.linalg.norm(photo_embeddings - query_embedding, axis=1)

            sorted_photos_embeddings = np.argsort(similares)
            top_10_predictions_per_sketch_index.append(sorted_photos_embeddings[:10])

        recall_precision_curve = compute_recall_precision_curve(top_10_predictions_per_sketch_index, sketch_labels, photo_labels, 10)

        recalls, precisions = zip(*recall_precision_curve)
        plt.plot(recalls, precisions, marker='o', linestyle='-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Recall-Precision Curve')
        plt.grid(True)
        plt.savefig('recall_precision.png')
        plt.show()


      
        _, ax = plt.subplots(10, 11)
        for i in range(10) :
            for j in range(11) :
                ax[i,j].set_axis_off()

        for n in range(10):
            print("Sketch: ", n)
            sketch_position = np.random.randint(len(sketch_embeddings))
            print("Obteniendo sketch: ", n)
            for j, sketch in enumerate(ds_test):
                if j == sketch_position:
                    sketch = sketch["sketch"]
                    break


            sketch_embedding = sketch_embeddings[sketch_position]
            similares = np.linalg.norm(photo_embeddings - sketch_embedding, axis=1)

            sorted_photos_embeddings = np.argsort(similares)
            closest_10_photos_index = sorted_photos_embeddings[:10]

            ax[n][0].imshow(sketch)
            photos = []
            print("Obteniendo foto: ", n)
            for j in range(1, 11):
                for k, sketch in enumerate(ds_test):
                  if k == closest_10_photos_index[j-1]:
                      photos.append(sketch["photo"])
                      break 
                ax[n][j].imshow(photos[j-1])
        plt.savefig('ejemplo.png')
        plt.show()



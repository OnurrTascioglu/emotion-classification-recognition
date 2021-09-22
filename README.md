## 1-PYTHON KODLARI 
Python kodları Google Colab üzerinde test edilmiş ve eğitim aşaması yapılmıştır. Kod detayları, açıklama satırı şeklinde belirtilmiştir. Python kodlarına aşağıdan ulaşabilirsiniz. 


<details>
<summary>Python Kodları!</summary>
  
```
from google.colab import drive  #Drive account
drive.mount('/content/gdrive')

!mkdir -p drive       
!google-drive-ocamlfuse drive

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf  #import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16
import cv2
# %matplotlib inline

import keras
from keras import backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras import models

root = '/content/gdrive/My Drive/Colab/YapaySinirAglari/' #Google Drive yolu belirtilmeli.

data = pd.read_csv(root + "data/fer2013/fer2013.csv") #fer2013 veri seti ve dataları Root dosya yolunun içinde olmalıdır.
data.shape

train_data = data[data.Usage == "Training"] #sadece eğitim örneklerini train_data değişkenine atanır

#eğitim örnekleri csv dosyası formatında her bir satırda bir fotoğrafın piksel değerleri olacak şekilde tablo halinde kaydedilmiştir. 
#bu piksel değerlerine erişmek için öncelikle parse edilmesi gerekir.
train_pixels = train_data.pixels.str.split(" ").tolist() 

train_pixels = pd.DataFrame(train_pixels, dtype=int)
train_images = train_pixels.values
train_images = train_images.astype(np.float) 

print(train_images)
print(train_images.shape)

#Eğitim görselleri 48x48 formatındadır.
def show(img):
    show_image = img.reshape(48,48)
    
    plt.axis('off') # görüntü çerçevesindeki pikselleri siler
    plt.imshow(show_image, cmap='gray')

train_labels_flat = train_data["emotion"].values.ravel()#dizideki emotion değerleri
                                                        #train_labels_flat değerine atanır.
train_labels_count = np.unique(train_labels_flat).shape[0]
#Eğitim esnasında training imagelerin sonuçlarının(duygularının) karşılaştırılması gerekir.
#Bu yüzden train_labels_flat değişkeninde bu değerler tutulur.
#Toplamda 7 farklı duygu olabilir ve her training image bir tane duygu değeri tutar

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0] #num_labels = toplam train image duyguları (train image sayısı kadar)
    index_offset = np.arange(num_labels) * num_classes #dizi elemanları 0*7,1*7,2*7,3*7 şeklinde ilerler
                                                       #Bunun amacı; train_image*duygu_sayısı bulmak ve satırları belirlemek
    labels_one_hot = np.zeros((num_labels, num_classes)) #bütün değerlere 0 atanır
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1 #index_offset satırı, labels_dense.ravel() stunu belirler
    #bütün fotoğrafların sahip olduğu emojiyi labels_one_hat dizisinde tutulur.
    
    return labels_one_hot

y_train = dense_to_one_hot(train_labels_flat, train_labels_count)
y_train = y_train.astype(np.uint8)
#dönen değer (28709, 7) elemanlı ve her satırda train image'in hangi duyguyu barındırdığını gösteren dizi
print(y_train.shape)

test_data = data[data.Usage == "PublicTest"] 
test_pixels = test_data.pixels.str.split(" ").tolist() #benzer şekilde test dataları image haline getirilir

test_pixels = pd.DataFrame(test_pixels, dtype=int)
test_images = test_pixels.values
test_images = test_images.astype(np.float)

print(test_images.shape)

test_labels_flat = test_data["emotion"].values.ravel()
test_labels_count = np.unique(test_labels_flat).shape[0]
#train imageler için yapılan işlemin aynısı test imagelerde tutulur.
y_test = dense_to_one_hot(test_labels_flat, test_labels_count)
y_test = y_test.astype(np.uint8)
#Test imageların etiketli olması başarı oranının hesaplanması için kaydedilir.

model = Sequential()

#model1 için örnek ağ yapısı ---------------------------------------------------------------------------------------
#Aynı anda yalnızca 2 modelden birinin eğitimi yapınız.

#
#model.add(Conv2D(64, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(48, 48, 1),use_bias = True, bias_initializer= "one"))
##ilk evrişim katmanı; 64 derinlikli, (3,3) maskeler ile filtrelenmiş, bias = 1 şeklindedir.
#model.add(BatchNormalization())
#model.add(Activation("relu"))
#model.add(MaxPool2D(pool_size=(2, 2), strides=2))
#model.add(Dropout(0.6)) #DropOut işlemi aşırı öğrenmeyi engellemek için ağırlıkların %60'ını unutur.
#
#model.add(Conv2D(32, 3, use_bias = True, bias_initializer= "one"))
#model.add(BatchNormalization())
#model.add(Activation("relu"))
#model.add(MaxPool2D(pool_size=(2, 2), strides=2))
#model.add(Dropout(0.6))
#
#model.add(Flatten()) #Tam bağlantılı katmana giriş
#model.add(Dense(128, use_bias = True, bias_initializer= "one"))       #1. tam bağlantılı katman
#model.add(BatchNormalization())
#model.add(Activation("relu"))
#model.add(Dropout(0.6))
#
#model.add(Dense(7))           #Çıkış katmanı
#model.add(Activation('softmax'))
#
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()



#--------------------------------------------------------------------------

#model2 için örnek ağ yapısı 

#-------------------------------------------------------------------------------------------------------
model.add(Conv2D(256, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(48, 48, 1),use_bias = True, bias_initializer= "one"))
#ilk evrişim katmanı; 256 derinlikli, (3,3) maskeler ile filtrelenmiş, bias = 1 şeklindedir.
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6)) #DropOut işlemi aşırı öğrenmeyi engellemek için ağırlıkların %60'ını unutur.


model.add(Conv2D(192, 3 ,use_bias = True, bias_initializer= "one"))  #2. katman
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.6))

model.add(Conv2D(128, 3 ,use_bias = True, bias_initializer= "one"))  #3. katman
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.6))


model.add(Conv2D(64, 3 ,use_bias = True, bias_initializer= "one"))   #4. katman
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.6))

model.add(Flatten()) #Tam bağlantılı katmana giriş
model.add(Dense(512,use_bias = True, bias_initializer= "one"))       #1. tam bağlantılı katman
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.6))

model.add(Dense(256,use_bias = True, bias_initializer= "one"))       #2. tam bağlantılı katman
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(7))           #Çıkış katmanı
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

x_train = train_images.reshape(-1, 48, 48, 1)
x_test = test_images.reshape(-1, 48, 48, 1)

print(y_test)

# en başarılı ağırlıkları kaydet
checkpointer = ModelCheckpoint(filepath=root + 'data/face_model.h5', verbose=1, save_best_only=True)

epochs = 10
batchSize = 128 

# modeli çalıştır
# x_train = Eğitim görüntüleri , x_test = Test görüntüleri
# y_train = Eğitim görüntülerinin duygu bilgisi , y_test = Test görüntülerinin duygu bilgisi
hist = model.fit(x_train, y_train, 
                 epochs=epochs,
                 shuffle=True,
                 batch_size=batchSize, 
                 validation_data=(x_test, y_test),
                 callbacks=[checkpointer], verbose=2)

#Model json tipinde kaydedilir.
model_json = model.to_json()
with open(root + "data/face_model.json", "w") as json_file:
    json_file.write(model_json)

#Ağırlık örnekleri
for layer in model.layers:
    if layer.get_weights() != []:
        print("Name:"+layer.name, layer.get_weights())

shape_dict = {} # (layer name:shape) save dictionary
loc = root + "data/" # save location

for layer in model.layers:
    if layer.get_weights() != []:
        shape_dict[layer.name] = np.shape(layer.get_weights()[0]) #Bias yok, sadece ağırlıklar kaydedilir (bias=1)
        np.savetxt(loc  + layer.name+ ".csv", layer.get_weights()[0].flatten() , delimiter=",") 
        with open(loc  + layer.name+ ".csv",'ab') as f: 
          np.savetxt(f, layer.get_weights()[1].flatten() , delimiter=",") # bias ağırlıkları kaydedilir
          if "batch_normalization" in layer.name:   #Batch 1. parametre =  γ parametre , Batch 2. parametre = β parametre
            np.savetxt(f, layer.get_weights()[2].flatten() , delimiter=",")  #Batch 3. parametre = aritmetik ortalama
            np.savetxt(f, layer.get_weights()[3].flatten() , delimiter=",")  #Batch 4. parametre = varyans

#with open(root + "data/face_model.txt", "w") as txt_file:
#    txt_file.write()

#Eğitimin görsel analizi
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Eğitim', fontsize=16)
plt.ylabel('Loss', fontsize=10)
plt.plot(hist.history['loss'], color='b', label='Training Loss')
plt.plot(hist.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=10)
plt.plot(hist.history['accuracy'], color='b', label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], color='r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

#private test için görüntüler ayrılır.
test = data[["emotion", "pixels"]][data["Usage"] == "PrivateTest"]
test["pixels"] = test["pixels"].apply(lambda im: np.fromstring(im, sep=' '))
test.head()

x_test_private = np.vstack(test["pixels"].values)
y_test_private = np.array(test["emotion"])

x_test_private = x_test_private.reshape(-1, 48, 48, 1)
y_test_private = np_utils.to_categorical(y_test_private)
x_test_private.shape, y_test_private.shape

score = model.evaluate(x_test_private, y_test_private, verbose=0)
print("PrivateTest üzerinde doğruluk başarımı:", score[1])

model.load_weights(root + 'data/face_model.h5')

#test_img_path = root + "images/fer1_images/sad1.jpg"
#
#img_orj = image.load_img(test_img_path)
#img = image.load_img(test_img_path, grayscale=True, target_size=(48, 48))
#
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis = 0)
# 
#custom = model.predict(x.reshape(-1, 48, 48, 1))


test_image=train_images[0].reshape( 48, 48)
custom = model.predict(test_image.reshape(-1, 48, 48, 1))

#1
objects = ('kızgın', 'nefret', 'korku', 'mutlu', 'üzgün', 'şaşırma', 'doğal')
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('duygu')

plt.show() 

#2

plt.axis('off')
plt.gray()
plt.imshow(test_image)

plt.show()
#------------------------------

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(test_image.reshape(1,48,48,1))
 
#katman çıkışlarını gözlemleyebilmek ve C++ karşılaştırmalarını yapabilmek için kullanılan fonksiyon. 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            print(activation[0, :, :, activation_index])
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

#Batch normalizasyonu parametrelerinin dense veya conv katmanlarının çıktılarının gözlemlebilmesini için batch normalizasyonu hesaplayan fonksiyon.
def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = np.mean(X, axis=0)
        # mini-batch variance
        variance = np.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / np.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, H, W, C = X.shape
        # mini-batch mean
        mean = np.mean(X, axis=(0, 1, 2))
        # mini-batch variance
        variance = np.mean((X - mean.reshape((1, 1, 1, C))) ** 2, axis=(0, 1, 2))
        # normalize
        X_hat = (X - mean.reshape((1, 1, 1, C))) * 1.0 / np.sqrt(variance.reshape((1, 1, 1, C)) + eps)
        # scale and shift
        out = gamma.reshape((1, 1, 1, C)) * X_hat + beta.reshape((1, 1, 1, C))

    return out

#1. katmanın batch normalizasyonu sonuçları
gamma = model.layers[1].get_weights()[0]
beta = model.layers[1].get_weights()[1]

pure_batch_norm(activations[0], gamma, beta, eps = 1e-5)

#15. satır (yani çıkış katmanı) sonuçlarını gösteren kod parçası

print(model.get_layer(index = 15).name)
activation = activations[15]
activation_index = 0;

for col in range(0,7):
    print(activation[0, activation_index])
    activation_index += 1

#2. evrişim katmanı  çıktılarını gösteren kod parçacığı
print(model.get_layer(index = 4).name)
display_activation(activations, 3, 2, 4)
  ```
</details>





## 2-EĞİTİM SETİ
- Eğitimde ve test aşamalarında kullanılan veri seti, 2013 yılında düzenlenen Facial Expression Recognition Challenge (fer2013) adlı yarışmada kullanılan veri setidir. Veri setinde 48x48 boyutunda gri görüntüler bulunmaktadır. Toplamda 35887 adet görüntü bulunmaktadır ve bu görüntülerden 28709 tanesi eğitim için kullanılmış olup, 7178 tanesi test için ayrılmıştır. Eğitim setinde duygular yedi sınıfa ayrılmıştır. Kızgın, nefret, korku, mutlu, üzgün, şaşkın, doğal.
- Eğitim setine, aşağıda belirtilen linkteki data sekmesi altında, icml_face_data.csv(287.1 MB) adlı dosya indirilerek ulaşılabilir. 
- Python kodlarında, eğitim setinin bulunduğu dosya konumu uygun şekilde (koddaki açıklama satırlarında belirtilen şekilde) belirtilmelidir.
- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv

## 3-ÖRNEK AĞIRLIKLAR
Örnek Ağırlıklar.zip dosyasında, pythonda eğitilen ağın ağırlıkları kaydedilmiştir, 1 adet örnek bulunmaktadır. Örnek model1'i temsil (model2 örnekleri yüksek dosya boyutlarına sahip olduğu için yüklenmedi) etmektedir ve agirliklar1 klasöründe model1'e ait bir örnek bulunmaktadır. “agirliklarData” klasörlerinin içerisinde örnek python çıktıları ve modele ait bilgiler mevcuttur. “.csv” uzantılı dosyalarda evrişim, tam bağlantılı ve normalizasyon ağırlıkları mevcuttur. “model.txt” dosyası modele ait katman bilgilerini içerir. Ağırlıklar okunmadan önce bu dosya okunur ve katman bilgileri kaydedilir. Gerçek zamanlı duygu analizi için örnek bir test videosu eklenmiştir. Model1 ve model2 şeması klasör içerisinde görsellerle gösterilmiştir. 
- Model.txt içeriği:
<br> ![image](https://user-images.githubusercontent.com/57223642/134393447-e6ee3896-3c41-48d1-a87a-11be4be298db.png) </br>

[Örnek Ağırlıklar.zip](https://github.com/OnurrTascioglu/emotion-classification-recognition/files/7212718/3.Ornek.Agirliklar.zip)

## 4- C++/CLR VE ARAÇLARIN KURULUMU
Projeyi çalıştırmak ve geliştirmek için Cuda ve OpenCV kurulması gerekir. Cuda, yapay sinir ağının grafik kartı üzerinde koşması için kurulmalıdır, OpenCV ise yüz algılama aşamasında kullanılmıştır. Bu araçların CLR projesi ile çalışabilmesi için bazı konfigürasyonlar gerekmektedir. Aşağıdaki videoları izleyerek bu konfigürasyonları gerçekleştirebilirsiniz.
- Cuda: https://www.youtube.com/watch?v=cuCWbztXk4Y
- OpenCV: https://www.youtube.com/watch?v=p-6rG6Zgu4U

## 5- C++ KODLARI

### 5.1- Ağırlıkların Dosyadan Okunması
Öncelikle ağırlıkların dosyadan okunması gerekir. Okunan ağırlıklar katman boyutlarına göre uygun dizilere aktarılmalıdır. Katman boyutları model.txt dosyasının içinde olmalıdır. Evrişim katmanında ağırlıklar aslında maskelerdir. Maskeler 3x3 boyutunda (değişebilir) ve ayrıca bir adet bias değerine sahiptir. Gizli katmanlarda maskelerin her biri, bir önceki katmanın maske sayısı kadar derinlik kazanır çünkü kendisinden önceki katman 2 boyutlu bir image değil, 3 boyutlu bir feature space dir. Bu işlemi yapan fonksiyonlar:
- readWeightFromFile(), weightsToolStripMenuItem_Click() 

### 5.2- Eğitim/Test Görüntülerinin Dosyadan Okunması
Test görüntüleri csv dosyası formatında tutulmuştur. Her satırda bir görüntünün duygu değeri, piksel değerleri ve ne amaçlı kullanılacağı bilgisi vardır. Bu dosya kullanılabilmesi için parse edilmesi gerekir. Bu işlemi yapan fonksiyon:
- fer2013DSToolStripMenuItem_Click()

### 5.3.1- Test Aşamasının CPU Üzerinde Koşulması
Öncelikle seçilen test görüntüsünün ekranda gösterilmesi için button1_Click adlı fonksiyon çalışır. Bu fonksiyon içinde çağrılan runToolStripMenuItem_Click fonksiyonu asıl hesaplamaları yapan fonksiyondur. runToolStripMenuItem_Click Fonksiyonunda evrişim, batch, reLU vb. işlemleri yapan fonksiyonlar sırası ile çağrılır. Katmanlardan önce bellek bölgeleri ve katman boyutları ayarlanır.

### 5.3.2- Test Aşaması Fonksiyonları
Test aşaması fonksiyonları Test.cpp dosyasında koşar.  Bu fonksiyonlar conv1(), convHidden(), batchNormalizationConv(), batchNormalizationDense(), reLU(), maxPooling(), flatten(), dense(), sofmax() fonksiyonlarıdır. 
- Conv1() fonksiyonu CNN katmanlardan giriş katmanını çağıran fonksiyondur. Bu fonksiyon giriş parametresi olarak 2 boyutlu bir görüntü(yükseklik ve genişlikle beraber) alır. Bu görüntüye evrişim işlemi uygular. Çıkış olarak 3 boyutlu feature space’i geri döndürür.
Pythonda giriş evrişim katmanı için ağırlıkların kaydedilme sırası soldaki şekilde verilmiştir. Bu dizilimin sağdaki şekilde yapılması daha  anlaşılır olmasını sağlar. (Dizilerde son elemandan sonra maskelerin bias değerleri sıralanır.)

![image](https://user-images.githubusercontent.com/57223642/134395499-1d74502e-b1d0-4ed9-b509-35ca4a13b84b.png)
&emsp; &emsp; &emsp; &emsp;
![image](https://user-images.githubusercontent.com/57223642/134395515-4d091a36-0365-4cfb-a12e-7e3eecfafe97.png)

Daha anlaşılır olması açısından aşağıdaki gifleri inceleyebilirsiniz.

- Python Ağırlıkları 
<br> ![Conv1InputWeights](https://user-images.githubusercontent.com/57223642/134395912-4f343d39-c0cf-499b-a335-79be1540d5af.gif) </br>

- Dönüştürülen Ağırlıklar 
<br> ![Conv1OutputWeights](https://user-images.githubusercontent.com/57223642/134395956-8ddd3060-5aef-42a6-8a6e-e70ee222f26b.gif) </br>

Bu işlemi yapan kod parçacığı aşağıda gösterilmiştir.
<br> ![image](https://user-images.githubusercontent.com/57223642/134396548-3cd4bec2-5bf4-47de-9126-5fb45ca528d2.png) </br>

- ConvHidden(), CNN katmanlardan gizli evrişim katmanları çağıran fonksiyondur. Bu fonksiyon giriş parametresi olarak 3 boyutlu bir feature space alır. Bu feature space’e evrişim işlemi uygular. 3 boyutlu uzaya uygulanan evrişim işleminde kullanılan maskeler de 3 boyutlu (derinlikli) olmalıdır. Bu derinlik önceki evrişim katmanının maske sayısı kadardır. Çıkış olarak yine 3 boyutlu bir feature space’i geri döndürür.
Benzer şekilde Pythonda gizli evrişim katmanı için ağırlıkların kaydedilme sırası soldaki şekilde verilmiştir. Bu dizilimin sağdaki şekilde yapılması daha anlaşılır olmasını sağlar. (Dizilerde son elemandan sonra maskelerin bias değerleri sıralanır.)

![image](https://user-images.githubusercontent.com/57223642/134397022-fc125623-e981-4303-b27d-b53b7cce67f7.png)
&emsp; &emsp; &emsp; &emsp;
![image](https://user-images.githubusercontent.com/57223642/134397044-0cf2279e-8bb0-46e0-91ae-65b220021f76.png)

Daha anlaşılır olması açısından aşağıdaki gifleri inceleyebilirsiniz.

- Python Ağırlıkları 
![convHiddenInputWeights](https://user-images.githubusercontent.com/57223642/134397298-206c16d0-24b3-4435-87fd-2f1924c4f55a.gif)

- Dönüştürülen Ağırlıklar 
![convHiddenOutputWeights gif](https://user-images.githubusercontent.com/57223642/134397316-87c9592e-bb12-4969-b6a4-dc1dc71f12bb.gif)

- Dense(), tam bağlantılı katmanlar için hesaplamaları yapan fonksiyondur. Bu fonksiyon giriş değerlerini parametre olarak alır. Bu giriş değerleri evrişim katmanının çıkışının flatten edilmiş değerleri olabileceği gibi başka bir dense katmanının çıkışı da olabilir. Ayrıca ağırlıkları ve giriş-çıkış katmanlarının boyutlarını da parametre olarak alır. Çıkış nöronlarının değerlerini geriye döndürür. Daha anlaşılır olması açısından aşağıdaki gifleri inceleyebilirsiniz.

- Python ağırlıkları kaydetme sırası
![denseanimation](https://user-images.githubusercontent.com/57223642/134397515-9341ad58-6b9d-4996-9c0f-e90928c9bc50.gif)


- Flatten() işlemi nöronlarla ağırlıkların düzgün sırada çarpılabilmesi için gerekli bir işlemdir. Bu işlem ile çıkış evrişim katmanı sıralanarak tam bağlantılı katmanın girişi olacaktır.
Girişte her bir feature, dizide sıralı şekilde tutulur. Bunu aşağıda soldaki şekilde görebilirsiniz. Flatten işleminden sonra sağdaki şekilde dizilmiş olur.

- Flatten Input
<br>  ![flattenInput](https://user-images.githubusercontent.com/57223642/134397749-25076484-f821-483e-bffb-af9fbacc1ade.gif)</br>

- Flatten Output
<br> ![flattenOutput](https://user-images.githubusercontent.com/57223642/134397835-01cd3d88-0b7b-4dc3-a3bd-546c84a7f0ba.gif) </br>



- BatchNormalizationConv() işlemi evrişim katmanında uygulanan batch Normalizasyon işlemi fonksiyonudur. Batch normalizasyon parametreleri sıralı şekilde bütün gammalar, betalar, aritmetik ortalamalar, varyanslar olacak şekilde sıralanmıştır. Bu Değerler eğitim aşamasında öğrenilir. Her feature için ayrı gamma, beta, aritmetik ortalama ve varyans değerleri vardır. 
- BatchNormalizationDense() işlemi tam bağlantılı katmanda uygulanan batch Normalizasyon işlemi fonksiyonudur. 
- reLU() işlemi sıfırdan küçük olan feature değerlerini sıfıra eşitler.
- maxPooling() işlemi havuzdaki en büyük değerin çıkışa aktarılmasıdır. Pool, havuz’un genişlik ve yükseliğini belirtirken, stride adım sayısını belirtir.
- Softmax() çıkıştaki sınıflandırmayı rasyonel değerlerden olasılık değerine dönüştürür.


## 6- GPU C++ ve CUDA KODLARI
### 6.1 CUDA nedir?
CUDA, Nvidia’nın sunduğu bir teknolojidir. C diliyle yazılmış algoritmaların, Nvidia grafik kartları üzerinde koşmasını sağlar. Bu çalışmadaki test aşaması Pascal mimarisine sahip, Nvidia Geforce GTX 1050 TI grafik kartında koşulmuştur. Grafik kartı üzerinde koşan kodların zamansal ve verimlilik açısından incelenmesi için CUDA’nın sunmuş olduğu Visual Profiler eklentisi kullanıldı. 
Pascal mimarili bir grafik kartında 60 adet Stream Multiprocessor (SM) bulunur. Her bir SM 2048 adet thread oluşturabilir. Threadler bloklar halinde SM üzerinde koşar. Tek bir SM üzerinde maksimum 32, minimum 2 thread bloğu oluşturulabilir. 32 thread bloğu oluşturulduğunda tam verimlilik için her bir blok üzerinde 64 (32*64=2048) thread koşar. 2 thread bloğu oluşturulduğu durumda ise tam verimlilik için her bir blok üzerinde 1024 (2*1024=2048) thread koşar. Toplamda 122.880 (60*2048) thread eş zamanlı koşabilir. Cuda hakkında daha detaylı bilgi edebilmek için aşağıdaki kursu takip edebilirsiniz.
- https://www.udemy.com/course/gpu-programlama/


### 6.2 Test Aşamasının GPU Üzerinde Koşulması
Kodların GPU ile koşabilmesi için, GPU ve CPU bellekleri arasında veri alışverişini sağlamalıyız. CPU ve GPU belleklerini işaret eden pointerları tutan bir struct yapısı inşa edilmeli. Bu işlemi gerçekleştiren fonksiyonlar:
- struct CpuGpuMem
### 6.3  GPU ve CPU Bellek İşlemleri
CPU GPU bellek işlemlerini gerçekleştiren fonksiyonlar CpuGpu.cpp dosyası içindedir. cpuGpuAlloc() fonksiyonu CPU ve GPU bellek alanlarını tahsis eder. İlk parametre olarak CpuGpuMem türü struct objesi alır. İkinci parametre, ayrılacak bellek alanının seçilmesi yarayan bir enum dır. Örnek olarak imageEnum, featureEnum, maskEnum vb. Üçüncü parametre olarak ayrılacak data tipinin boyutu (sizeOfType) belirtilir. İstenen boyuta göre hem CPU hem GPU bellek bölgelerini tahsis eder.
cpuGpuFree(), tahsis edilen bellek bölgelerini serbest bırakan fonksiyondur. cpuGpuAlloc() gibienum parametresi alır.
cpuGpuPin() ve cpuGpuUnPin() GPU belleği üzerinde belirli bir bellek alanının pinlenmesini sağlar. Bu pinleme işlemi GPU’nun pinlenmiş bölgeye daha hızlı erişmesini sağlar. Fakat pinleme işlemi bellekte fragmentasyon işlemine sebep olabilir.
cpuGpuMemCopy(), GPU ve CPU bellek bölgeleri arasında data transferi yapılmasını sağlar. Enum olarak belirlenmiş parametre ile GPU’dan CPU’ya veya CPU’dan GPU’ya veri transferini yapar.

### 6.4  Model 1 in GPU ile Koşması
Modelin GPU üzerinde koşaması birinci evirşim katmanı (conv1) ile başlar. Evrişim işlemi öncesinde bazı bellek bölgelerinin tahsis edilmesi ve parametrelerin (feature boyutları, maske boyutları, pool, stride gibi değerler) belirlenmesi gerekir. Bu işlemler setValuesForGpuConv1() fonksiyonunda yapılır. Aynı zamanda dosyadan okunmuş ağırlıklar struct yapısı olarak tanımlanmış CPU bellek bölgelerine kopyalanır. Ardından gerekli datalar CPU bellek bölgesinden GPU bellek bölgesine transfer edilir conv1ExecGPU() fonksiyonu çağrılır (ExecGPU fonksiyonları 6.5 konu başlığı altında anlatılmıştır). Sonrasında geriye dönen dizi birinci evrişim katmanının çıkış uzayıdır. Çıkış katmanını görsel olarak inceleyebilmek için showFeatureOnPictureBox() fonksiyonu kullanılır. İkinci evrişim katmanından önce benzer şekilde bazı bellek bölgelerinin tahsis edilmesi ve parametrelerin belirlenmesi gerekir. setValuesForGpuConv2() fonksiyonu bu işlemleri yapar. Ardından gerekli data transferleri gerçekleşir ve convHidden1ExecGPU() fonksiyonu ile ikinci evrişim katmanı koşulur. Dense katmanları için de benzer işlemler tekrarlanır. Son olarak Softmax işlemi yapılır.
Bu işlemi gerçekleştiren fonksiyonlar MyForm.h dosyası içinde mevcuttur.

### 6.5  Model 1 GPU Fonksiyonları
Cuda fonksiyonları __global__, __device__, __host__ vb. ön ekleriyle tanımlanır. Bu fonksiyonlar sadece .cu uzantılı dosyalarda derlenebilir. Cuda fonksiyonlarının C/C++ dilinde koşabilmesi için, köprü fonksiyonlara ihtiyacı vardır. Bu köprü fonksiyonlar aşağı verilmiştir.
- conv1ExecGPU(), convHidden1ExecGPU(), dense1ExecGPU(), dense2ExecGPU() ;
Köprü fonksiyonlarda thread blokları ve sayıları belirlenir ve sonrasında __global__  fonksiyonlar çağrılır. conv1ExecGPU() köprü fonksiyonunda birinci evrişim katmanında yapılacak bütün işlemler sıralanır. Benzer şekilde diğer köprü fonksiyonlarda da bütün işlemler sıralanır. 
__global__ fonksiyonlar GPU üzerinde koşan fonksiyonlardır.  Global fonksiyonlar conv1GPU << <gridDim, blockDim >> > (a,b,c) şeklinde çağrılır. conv1GPU fonksiyon adıdır gridDim thread bloklarının sayısını, blockDim ise her bloktaki thread sayısını temsil eder. Ardından fonksiyon parametreleri (a, b, c) sıralanır. Benzer işlemler Model 2 fonksiyonlarında kullanılmıştır.
Bu işlemi gerçekleştiren fonksiyonlar Test.cu dosyası içinde mevcuttur. 

### 6.6  Yüzün Algılanması 
Yüzün algılanmasında openCV haarcascade kullanılmıştır. Bu işlemi gerçekleştiren fonksiyonlar MyForm.h dosyasındaki detectAndDraw(), printFaces() fonksiyonlarıdır.

### 6.7  Video ve Kameradaki Görüntü Üzerinde Duygu Analizi
Video ve kameradaki görüntü üzerinde duygu analizi işlemini gerçekleştiren fonksiyonlar MyForm.h dosyasındaki button3_Click(), button4_Click() fonksiyonlarıdır





 		









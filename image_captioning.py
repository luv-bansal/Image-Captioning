import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense,Embedding,Input,Flatten,GRU
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences     

import csv
encoder_pre_model = VGG16(include_top=True, weights='imagenet')

encoder_output_layer = encoder_pre_model.get_layer('fc2')
encoder_output = encoder_output_layer.output
encoder_output = Dense(256, activation=tf.nn.tanh)(encoder_output)

encoder_model = Model(encoder_pre_model.input, encoder_output)
encoder_model = Model(encoder_pre_model.input, encoder_output)

for n in range(len(encoder_model.layers)):
    encoder_model.layers[n-1].trainable = False
encoder_model.summary()
image_dir = r'C:\ml\image captioning\31296_39911_bundle_archive\flickr30k_images\flickr30k_images'
caption_csv = r'C:\ml\image captioning\31296_39911_bundle_archive\flickr30k_images\results.csv'
imgcap = {}
captions = []
uni_img = []
uni_cap = []
c=0
d=0
with open(caption_csv, 'r',encoding="utf8") as file:
    
    reader = csv.reader(file, delimiter='|')
    next(reader)
    for row in reader:
        
        try:
            imgcap[row[0] + ' ' + row[1]] = row[2]
            captions.append(row[2])
            if row[0] in uni_img:
                
                pass
            else:
                uni_img.append(row[0])
        except UnicodeDecodeError:
            
            d+=1
        except IndexError:
            c+=1

## ---(Sun Aug 16 02:56:21 2020)---

encoder_pre_model = VGG16(include_top=True, weights='imagenet')

encoder_output_layer = encoder_pre_model.get_layer('fc2')
encoder_output = encoder_output_layer.output
encoder_output = Dense(256, activation=tf.nn.tanh)(encoder_output)

#Encoder Model
encoder_model = Model(encoder_pre_model.input, encoder_output)
for n in range(len(encoder_model.layers)):
    encoder_model.layers[n].trainable = False

encoder_model.summary()
image_dir = r'C:\ml\image captioning\31296_39911_bundle_archive\flickr30k_images\flickr30k_images'
caption_csv = r'C:\ml\image captioning\31296_39911_bundle_archive\flickr30k_images\results.csv'
imgcap = {}
captions = []
uni_img = []
uni_cap = []
c=0
d=0
with open(caption_csv, 'r',encoding="utf8") as file:
    
    reader = csv.reader(file, delimiter='|')
    next(reader)
    for row in reader:
        
        try:
            imgcap[row[0] + ' ' + row[1]] = row[2]
            captions.append(row[2])
            if row[0] in uni_img:
                
                pass
            else:
                uni_img.append(row[0])
        except UnicodeDecodeError:
            
            d+=1
        except IndexError:
            c+=1

len(uni_img)

for image in uni_img:
    
    uni_cap.append(imgcap[image+'  0'])
images = []
for img in list(imgcap.keys()):
    
    images.append(img)

import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
def load_image(image_name):
  image_file=os.path.join(image_dir,image_name)
  image = plt.imread(image_file)
  image=resize(image,(224,224) )
  return image

training_dir= r'C:\ml\image captioning\31296_39911_bundle_archive\flickr30k_images'
def mark_captions(captions):
  new_captions=[]
  for cap in captions:
    new_cap='sss ' + cap +' eee'
    new_captions.append(new_cap)
  return new_captions

mark_uni_cap=mark_captions(uni_cap)

class TokenizerWrap(Tokenizer):
    
    def __init__(self, texts):
        
        Tokenizer.__init__(self, num_words=10000)
        
        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)
        
        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),self.word_index.keys()))
    def token_to_string(self,tokens_s):
        
        string=[]
        tokens=tokens_s.split()
        for token in tokens:
            string.append(self.index_to_word[token])
        string=' '.join(string)
        
        return string
tokenizer=TokenizerWrap(mark_uni_cap)
token_uni_string=tokenizer.texts_to_sequences(mark_uni_cap)
decoder_max_len=max([len(i) for i in mark_uni_cap])
decoder_data=pad_sequences(token_uni_string,maxlen=decoder_max_len,padding='post',truncating='post')
decoder_data=list(decoder_data)
word_index=tokenizer.word_index
word_index=tokenizer.word_index
embedding_dim=100
embeddings_index = {}
vocab_size=len(word_index)
with open(r'C:\ml\image captioning\31296_39911_bundle_archive\glove.6B.100d.txt',encoding="utf8") as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
decoder_state=encoder_model.get_layer('dense')
decoder_initial_state=decoder_state.output
decoder_input=Input(shape=(None,),name='decoder_input')
decoder_embedding=Embedding(vocab_size+1, embedding_dim, input_length=decoder_max_len-1, weights=[embeddings_matrix], trainable=False)(decoder_input)
decoder_gru1=GRU(256,return_sequences = True)(decoder_embedding,initial_state=decoder_initial_state)
decoder_gru3=GRU(256,return_sequences = True)(decoder_gru1,initial_state=decoder_initial_state)
decoder_output=Dense(10000,activation='softmax',name='decoder_output')(decoder_gru3)
encoder_input=encoder_model.input
model_train=Model([encoder_input,decoder_input],[decoder_output])

encoder_input=model_train.get_layer('input_1')
encoder_input._name='encoder_input'
model_train.summary()
model_train.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(lr=1e-3),metrics=['accuracy'])
path_checkpoint = r'C:\ml\image captioning\31296_39911_bundle_archive\checkpoint_weights'
callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                         verbose=1,
                                                         save_weights_only=True)

callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir='C:\ml\image captioning\31296_39911_bundle_archive\callback_tensor',
                                                      histogram_freq=0,
                                                      write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator=ImageDataGenerator(rescale=1/255)
generator_train=generator.flow_from_directory(directory=training_dir,batch_size=32,target_size=(224, 224),class_mode = None)

len(os.listdir(r'C:\ml\image captioning\31296_39911_bundle_archive\flickr30k_images\flickr30k_images'))
def get_random_caption_tokens(idx):
  idx=list(idx)
  decoder_batch_data=[]
  for i in idx:
    token=decoder_data[i]
    decoder_batch_data.append(token)
  return decoder_batch_data
def batch_generator(batch_size):
  start=0
  stop=batch_size
  while True:
    idx= np.arange(start,stop)
    decoder_batch_data = get_random_caption_tokens(idx)
    decoder_batch_data=np.array(decoder_batch_data,dtype=float)
    decoder_input_data=np.array(decoder_batch_data[:,:-1],dtype=float)
    decoder_output_data=np.array(decoder_batch_data[:,1:],dtype=float)
    decoder_output_data=np.expand_dims(decoder_output_data, -1)
    
    x_data = \
    {
        'decoder_input': decoder_input_data,
        'encoder_input': generator_train.next()
    }
    y_data = \
    {
        'decoder_output': decoder_output_data
    }
    
    start+=batch_size
    stop+=batch_size
    yield (x_data,y_data)
generator_decoder = batch_generator(batch_size=32)

model_train.load_weights(path_checkpoint)

history=model_train.fit(x=generator_decoder,epochs=1,callbacks=callbacks,steps_per_epoch=1)

def generate_caption(image_name, max_tokens=30):
  image = load_image(image_name)
  image = np.expand_dims(image, axis=0)
  image=image.astype(np.uint8)
  image=image/255
  shape = (1, max_tokens)
  decoder_input_data = np.zeros(shape=shape, dtype=np.int)
  token_int = 2
  output_text = ''
  count_tokens = 0
  
  while (token_int != 3) and( count_tokens < max_tokens):
    decoder_input_data[0, count_tokens] = token_int
    x_data = \
    {
        'decoder_input': decoder_input_data,
        'encoder_input': image
    
    }
    output = model_train.predict(x_data)
    token_onehot = output[0, count_tokens, :]
    token_int = np.argmax(token_onehot)
    if token_int !=0:
        sampled_word = tokenizer.index_to_word[token_int]
    else:
        sampled_word=' '
    output_text  = output_text+ ( " " + sampled_word)
    count_tokens = count_tokens + 1
  print("Predicted caption:")
  print(output_text)
generate_caption('1000092795.jpg', max_tokens=30 )
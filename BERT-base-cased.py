
import numpy as np 
import pandas as pd 

def get_annotated(row):
    row['Annotation-Level1'] = row['Annotation-Level1'].strip()
    row['Annotation-Level2'] = row['Annotation-Level2'].strip()
    if (row['Annotation-Level1'],row['Annotation-Level2']) == ("Threat","Sexist"):
        return 0
    elif (row['Annotation-Level1'],row['Annotation-Level2']) == ("Threat","Non-Sexist"):
        return 1
    elif (row['Annotation-Level1'],row['Annotation-Level2']) == ("Non-Threat","Sexist"):
        return 2
    elif (row['Annotation-Level1'],row['Annotation-Level2']) == ("Non-Threat","Non-Sexist"):
        return 3
    return -999

def shuffle(df):
    shuffled = df.sample(frac=1)
    return shuffled
df = pd.read_csv('../input/threatdetection/ThreatDataset_modified (1).csv')
df['Label'] = df.apply (lambda row: get_annotated(row), axis=1)
df = shuffle(df)
#print(df.head())

#Splitting data into test and train datasets
train_size = int(len(df)*0.8)
train_df = df[:train_size]
test_df = df[train_size:]
test_df = test_df.dropna()
train_df = train_df.dropna()

from tensorflow.keras.utils import to_categorical

#converting our integer coded Labels column into categorical data(matrix)
y_train = to_categorical(train_df.Label,6)
y_test = to_categorical(test_df.Label,6)
#print(y_train)

from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', use_fast=False)
bert = TFBertModel.from_pretrained('bert-base-cased')

#Tokenising input from BERT base cased
x_train = tokenizer(
    text=train_df.Text.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True
    )
x_test = tokenizer(
    text=test_df.Text.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

input_ids = x_train['input_ids']
attention_mask = x_train['attention_mask']

#Building the Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense

max_len = 70
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(6,activation = 'sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()
#Model Testing and Evaluation
train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
    ),
  epochs=5,
    callbacks=[early_stopping]
)

predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})
print(predicted_raw[0])

y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = test_df.Label

from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))

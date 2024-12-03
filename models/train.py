from transformers import BertTokenizer, TFBertForSequenceClassification, AdamWeightDecay
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def load_dataset():
  dataset, info = tfds.load("goemotions", with_info=True)
  train_dataset = dataset["train"]
  val_dataset = dataset["validation"]
  test_dataset = dataset["test"]
  sentiment_keys = [key for key in info.features.keys() if isinstance(info.features[key], tfds.features.ClassLabel) or key not in ["comment_text"]]
  return train_dataset, val_dataset, test_dataset, sentiment_keys

def preprocess_dataset(dataset, sentiment_keys, max_length=128, batch_size=32):
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  def encode(text, labels):
    text = text.numpy().decode("utf-8")
    labels = labels.numpy() 
    tokens = tokenizer(
      text,
      max_length=max_length,
      padding="max_length",
      truncation=True,
      return_tensors="tf"
    )
    return tokens["input_ids"][0], tokens["attention_mask"][0], labels

  def map_func(data):
    text = data["comment_text"]
    labels = tf.stack([tf.cast(data[key], tf.float32) for key in sentiment_keys], axis=-1)
    input_ids, attention_mask, labels = tf.py_function(
      func=encode,
      inp=[text, labels],
      Tout=(tf.int32, tf.int32, tf.float32)
    )

    input_ids.set_shape([max_length])
    attention_mask.set_shape([max_length])
    labels.set_shape([len(sentiment_keys)])
    return {"input_ids": input_ids, "attention_mask": attention_mask}, labels

  dataset = dataset.map(map_func)
  dataset = dataset.cache()
  dataset = dataset.shuffle(batch_size * 10).batch(batch_size)
  dataset = dataset.prefetch(tf.data.AUTOTUNE)
  return dataset

def train_model(train_dataset, val_dataset, num_labels, epochs=3):
  model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
  )

  optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

  model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

  return model


def save_model(model, model_path):
  model.save_pretrained(model_path)

def evaluate_model(model, test_dataset):
  loss, accuracy = model.evaluate(test_dataset)
  print(f"Loss: {loss}")
  print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
  print("Loading dataset...")
  train_dataset, val_dataset, test_dataset, sentiment_keys = load_dataset()
  print("Sentiment Keys: ", sentiment_keys)
  print("Preprocessing dataset...")
  
  train_dataset = preprocess_dataset(train_dataset, sentiment_keys)
  val_dataset = preprocess_dataset(val_dataset, sentiment_keys)
  test_dataset = preprocess_dataset(test_dataset, sentiment_keys)

  print("Training model...")
  num_labels = len(sentiment_keys)
  print(f"Num Labels: {num_labels}")
  model = train_model(train_dataset, val_dataset, num_labels)

  print("Evaluating model...")
  evaluate_model(model, test_dataset)

  print("Saving model...")
  model_path = "model/sentiment_model"
  save_model(model, model_path)


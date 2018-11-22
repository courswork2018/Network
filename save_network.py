from keras.models import model_from_json
import numpy
from keras.preprocessing import sequence
import keras.utils

print("Загружаю сеть из файлов")
# Загружаем данные об архитектуре сети
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель
loaded_model = model_from_json(loaded_model_json)
# Загружаем сохраненные веса в модель
loaded_model.load_weights("mnist_model.h5")
print("Загрузка сети завершена")

max_review_length = 250
text = numpy.array(['Скажу честно — ожидала большего! Никаких чувств и эмоций после просмотра особо не было.'])
#print(text.shape)
tk = keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
tk.fit_on_texts(text)
prediction = loaded_model.predict(sequence.pad_sequences(tk.texts_to_sequences(text),maxlen=max_review_length))
print(prediction)
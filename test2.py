
import csv
import os
import tensorflow as tf
import keras
from keras import layers
import sklearn.model_selection

tf.config.set_visible_devices([], 'GPU')

# User ID,Name,Email,Preferred Activities,Bucket list destinations Sri Lanka
# 1,Jennifer Quinn,jennifer.quinn@example.com,"['cycling', 'historical monuments', 'village homestays']","['Polonnaruwa', 'Hatton', 'Anuradhapura', 'Ella', 'Haputale']"

# name,lat,lng,formatted_address,rating,user_ratings_total,latest_reviews
# Arugam Bay Beach,6.840407799999999,81.8368478,"Arugam Bay Beach, Sri Lanka",4.8,1591,"['Arugam Bay Beach is a surfer's paradise! I spent incredible days riding the waves, and the local surf schools were fantastic for beginners like me. The atmosphere is laid-back, with friendly locals and fellow travelers. After a long day of surfing, the sunsets were simply magical. The beach is a bit crowded, especially during peak season, but it adds to the lively vibe. I canÃ¢Â€Â™t wait to return!', 'My friends and I had an unforgettable time at Arugam Bay Beach! The surfing conditions were excellent, and we all managed to catch some great waves. The beach is beautiful, with soft sand and clear waters perfect for swimming. However, we noticed some litter on the beach, which was a bit disappointing. Overall, the vibrant nightlife and delicious food made up for it. Definitely worth a visit!', 'As a couple looking for relaxation, Arugam Bay Beach offered a perfect blend of tranquility and excitement. We enjoyed lazy days lounging on the beach and indulging in fresh seafood at beachside restaurants. While the surf scene was lively, it was easy to find quieter spots to unwind. The only downside was the occasional noise from nearby parties, but it didnÃ¢Â€Â™t detract much from our experience. A lovely getaway!', 'I visited Arugam Bay Beach with my family, and while the children loved the surf lessons, I found the beach a bit overcrowded. The atmosphere was vibrant, and the locals were warm and welcoming. We spent some time exploring nearby attractions like Elephant Rock, which was a highlight. Just wish there were more efforts to keep the beach clean as it detracted from the beauty. Overall, a memorable trip!', 'Arugam Bay Beach has its charm but also its downsides. The surfing was fantastic, and I managed to improve my skills with the help of local instructors. However, I was disappointed by the litter scattered along the beach. ItÃ¢Â€Â™s a shame because the natural beauty is stunning. The cafes and restaurants are great, but I believe more attention should be given to maintaining the beach. I enjoyed my time overall but hope for improvements in the future.']"

u = csv.DictReader(open('u.csv'))
# print(u.fieldnames)

xs = []
for x in u:
    a = eval(x['Preferred Activities'])
    d = eval(x['Bucket list destinations Sri Lanka'])
    # print(a, d)
    xs.append((a, d))

# xs = xs[:10]
xs = xs[:1]

ks = list(set([x for a, d in xs for x in a]))
ks.sort()
vs = list(set([x for a, d in xs for x in d]))
vs.sort()
print(len(xs), len(ks), len(vs))


model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(len(ks),)))
model.add(layers.Dense(units=len(vs), use_bias=False, kernel_initializer='zeros'))
# model.add(layers.Reshape(target_shape=(len(ks), 1)))
# model.add()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

print(model.get_weights())
exit()


fp = 'data/y.h5'


def train():
    def gen(xs, z):
        while 1:
            for a, d in xs:
                i = [1 if k in a else 0 for k in ks]
                for dd in d:
                    o = [1 if v == dd else 0 for v in vs]
                    yield tf.convert_to_tensor([i]), tf.convert_to_tensor(o)
            if z:
                break

    # traind, testd = sklearn.model_selection.train_test_split(xs, test_size=.1)
    traind = xs

    try:
        os.remove('logs')
    except:
        pass
    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1, update_freq=100)

    model.fit(gen(traind, 0), steps_per_epoch=len(traind), epochs=1000, callbacks=[tensorboard_callback])
    # model.fit(gen(traind, 0), validation_data=gen(testd, 1), steps_per_epoch=len(traind), epochs=1)
    # print(model.evaluate(gen(testd, 1)))

    # print(model(tf.ones(shape=(1, len(ks)))))
    model.save_weights(fp)


train()
model.load_weights(fp)

i = ['hot springs']
i = [1 if k in i else 0 for k in ks]

o = model.call(inputs=tf.convert_to_tensor([i]))
o = tf.math.top_k(o, k=5)
print(o.values.numpy().tolist()[0])
o = o.indices.numpy().tolist()[0]
o = [vs[i] for i in o]

print(o)

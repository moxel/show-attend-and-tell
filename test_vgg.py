import moxel
# Does it feel better to be "moxel.Image"
from moxel.space import Image

model = moxel.Model('strin/vgg19:latest', where='localhost')

# Ping the model to see if it works.
ok = model.ping()
print('ok', ok)

# Make a prediction.
img = Image.from_file('example/rock.jpg')
results = model.predict({
    'image': img
})


print(results['feature'])

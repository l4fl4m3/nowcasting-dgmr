import sonnet as snt
import discriminator
import generator
import tensorflow.compat.v1 as tf

class DGMR(snt.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.generator = generator.Generator(lead_time=90, time_delta=5)
    self.discriminator = discriminator.Discriminator()

  def generate(self, inputs, is_training=True):
    return self.generator(inputs, is_training=is_training)

  def discriminate(self, inputs, is_training=True):
    return self.discriminator(inputs, is_training=is_training)
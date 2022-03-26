import tensorflow.compat.v1 as tf
import sonnet as snt

def loss_hinge_disc(score_generated, score_real):
  """Discriminator hinge loss."""
  l1 = tf.nn.relu(1. - score_real)
  loss = tf.reduce_mean(l1)
  l2 = tf.nn.relu(1. + score_generated)
  loss += tf.reduce_mean(l2)
  return loss


def loss_hinge_gen(score_generated):
  """Generator hinge loss."""
  loss = -tf.reduce_mean(score_generated)
  return loss


def grid_cell_regularizer(generated_samples, batch_targets):
  """Grid cell regularizer.

  Args:
    generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
    batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

  Returns:
    loss: A tensor of shape [batch_size].
  """
  gen_mean = tf.reduce_mean(generated_samples, axis=0)
  weights = tf.clip_by_value(batch_targets, 0.0, 24.0)
  loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets) * weights)
  return loss

class Optimizer(snt.Module):

  def __init__(self,
               gan,
               gen_batch_size=2,
               disc_lr=2E-4,
               gen_lr=5E-5,
               disc_beta1=0.0,
               disc_beta2=0.999,
               gen_beta1=0.0,
               gen_beta2=0.999,
               loss_type='hinge',
               num_epochs=100,
               decay_lr_start_epoch=50,
               decay_disc_lr=True,
               decay_gen_lr=True,
               name=None):
    super().__init__(name=name)
    self.gan = gan
    self.gen_batch_size = gen_batch_size
    self.init_disc_lr = disc_lr
    self.init_gen_lr = gen_lr
    self.disc_beta1 = disc_beta1
    self.disc_beta2 = disc_beta2
    self.gen_beta1 = gen_beta1
    self.gen_beta2 = gen_beta2
    self.disc_lr = tf.Variable(
        disc_lr, trainable=False, name='disc_lr', dtype=tf.float32)
    self.gen_lr = tf.Variable(
        gen_lr, trainable=False, name='gen_lr', dtype=tf.float32)
    self.disc_opt = snt.optimizers.Adam(learning_rate=self.disc_lr, beta1=self.disc_beta1, beta2=self.disc_beta2)
    self.gen_opt = snt.optimizers.Adam(learning_rate=self.gen_lr, beta1=self.gen_beta1, beta2=self.gen_beta2)
    self.num_epochs = tf.constant(num_epochs, dtype=tf.int32)
    self.decay_lr_start_epoch = tf.constant(decay_lr_start_epoch, dtype=tf.int32)
    self.decay_disc_lr = decay_disc_lr
    self.decay_gen_lr = decay_gen_lr

  '''
  def disc_step(self, batch_inputs, batch_targets, lr_mult=1.):
    """Updates the discriminator once on the given batch of (images, labels)."""
    gan = self.gan
    with tf.GradientTape() as tape:
      batch_predictions = gan.generate(batch_inputs)
      gen_sequence = tf.concat([batch_inputs, batch_predictions], axis=1)
      real_sequence = tf.concat([batch_inputs, batch_targets], axis=1)
      concat_inputs = tf.concat([real_sequence, gen_sequence], axis=0)
      concat_outputs = gan.discriminate(concat_inputs)
      score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
      loss = loss_hinge_disc(score_generated, score_real)
    disc_params = gan.discriminator.trainable_variables
    disc_grads = tape.gradient(loss, disc_params)
    if self.decay_disc_lr:
      self.disc_lr.assign(self.init_disc_lr * lr_mult)
    self.disc_opt.apply(disc_grads, disc_params)
    #print("COMPLETED DISC STEP")
    return loss

  def gen_step(self, batch_inputs, batch_targets, lr_mult=1.):
    """Updates the generator once."""
    gan = self.gan
    num_samples_per_input = 6
    with tf.GradientTape() as tape:
      gen_samples = [ gan.generate(batch_inputs) for _ in range(num_samples_per_input)]
      grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),batch_targets)
      gen_sequences = [tf.concat([batch_inputs, x], axis=1) for x in gen_samples]
      gen_disc_loss = loss_hinge_gen(tf.concat(gen_sequences, axis=0))
      loss = gen_disc_loss + 20.0 * grid_cell_reg
    gen_params = gan.generator.trainable_variables
    gen_grads = tape.gradient(loss, gen_params)
    if self.decay_gen_lr:
      self.gen_lr.assign(self.init_gen_lr * lr_mult)
    self.gen_opt.apply(gen_grads, gen_params)
    #print("COMPLETED GEN STEP")
    return loss

  def _get_lr_mult(self, epoch=1):
    # Linear decay to 0.
    decay_epoch = tf.cast(epoch - self.decay_lr_start_epoch, tf.float32)
    if decay_epoch < tf.constant(0, dtype=tf.float32):
      return tf.constant(1., dtype=tf.float32)
    num_decay_epochs = tf.cast(self.num_epochs - self.decay_lr_start_epoch,
                               dtype=tf.float32)
    return (num_decay_epochs - decay_epoch) / num_decay_epochs

  def step(self, train_batches, epoch):
    """Updates the discriminator and generator weights.

    The discriminator is updated `len(train_batches)` times and the generator is
    updated once.

    Args:
      train_batches: list of batches, where each item is an (image, label)
        tuple. The discriminator is updated on each of these batches.
      epoch: the epoch number, used to decide the learning rate multiplier for
        learning rate decay.

    Returns:
      loss: the generator loss.
      lr_mult: the computed learning rate multiplier.
    """
    
    lr_mult = self._get_lr_mult(epoch)
    print(f'len TB: {len(train_batches)}')
    for train_batch in train_batches:
      self.disc_step(*train_batch, lr_mult=lr_mult)
    return self.gen_step(*train_batches[-1],lr_mult=lr_mult), lr_mult
    '''
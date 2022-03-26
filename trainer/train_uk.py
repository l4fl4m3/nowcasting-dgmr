from optimizer import Optimizer
from dgmr import DGMR
import tensorflow.compat.v1 as tf
import os
import tqdm
import sonnet as snt


#-----------------------------------------------------------------------------------
#FOR TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
#tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = snt.distribute.TpuReplicator(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
#------------------------------------------------------------------------------------


EXAMPLE_DATASET_BUCKET_PATH = "gs://dm-nowcasting/datasets/nowcasting_open_source_osgb/nimrod_osgb_1000m_yearly_splits/radar/20200718"
DATASET_ROOT_DIR = EXAMPLE_DATASET_BUCKET_PATH



_FEATURES = {name: tf.io.FixedLenFeature([], dtype)
             for name, dtype in [
               ("radar", tf.string), ("sample_prob", tf.float32),
               ("osgb_extent_top", tf.int64), ("osgb_extent_left", tf.int64),
               ("osgb_extent_bottom", tf.int64), ("osgb_extent_right", tf.int64),
               ("end_time_timestamp", tf.int64),
             ]}

_SHAPE_BY_SPLIT_VARIANT = {
    ("train", "random_crops_256"): (24, 256, 256, 1),
    ("valid", "subsampled_tiles_256_20min_stride"): (24, 256, 256, 1),
    ("test", "full_frame_20min_stride"): (24, 1536, 1280, 1),
    ("test", "subsampled_overlapping_padded_tiles_512_20min_stride"): (24, 512, 512, 1),
}

_MM_PER_HOUR_INCREMENT = 1/32.
_MAX_MM_PER_HOUR = 128.
_INT16_MASK_VALUE = -1

def parse_and_preprocess_row(row, split, variant):
  result = tf.io.parse_example(row, _FEATURES)
  shape = _SHAPE_BY_SPLIT_VARIANT[(split, variant)]
  radar_bytes = result.pop("radar")
  radar_int16 = tf.reshape(tf.io.decode_raw(radar_bytes, tf.int16), shape)
  mask = tf.not_equal(radar_int16, _INT16_MASK_VALUE)
  radar = tf.cast(radar_int16, tf.float32) * _MM_PER_HOUR_INCREMENT
  radar = tf.clip_by_value(
      radar, _INT16_MASK_VALUE * _MM_PER_HOUR_INCREMENT, _MAX_MM_PER_HOUR)
  result["radar_frames"] = radar
  result["radar_mask"] = mask
  return result


def reader(split="train", variant="random_crops_256", batch_size=16, shuffle_files=False):
  """Reader for open-source nowcasting datasets."""
  
  shards_glob = os.path.join(DATASET_ROOT_DIR, split, variant, "*.tfrecord.gz")
  shard_paths = tf.io.gfile.glob(shards_glob)
  shards_dataset = tf.data.Dataset.from_tensor_slices(shard_paths)
  if shuffle_files:
    shards_dataset = shards_dataset.shuffle(buffer_size=len(shard_paths))
  return (
      shards_dataset
      .interleave(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
                  num_parallel_calls=tf.data.AUTOTUNE,
                  deterministic=not shuffle_files)
      .map(lambda row: parse_and_preprocess_row(row, split, variant),
           num_parallel_calls=tf.data.AUTOTUNE)
      # Do your own subsequent repeat, shuffle, batch, prefetch etc as required.
      .shuffle(buffer_size=100)
      .repeat()
      .batch(batch_size, drop_remainder=True)
      #.prefetch(8)

  )

def get_dataset(batch_size):
  return reader(split="train", variant="random_crops_256", batch_size=batch_size)

                                           
num_epochs = 5
num_disc_steps = 2
batch_size = 8

# Calculate per replica batch size, and distribute the datasets on each TPU
# worker.
per_replica_batch_size = batch_size // strategy.num_replicas_in_sync

train_dataset = strategy.experimental_distribute_datasets_from_function(
    lambda _: get_dataset(per_replica_batch_size))

# Create the model, optimizer and metrics inside the strategy scope, so that the
# variables can be mirrored on each device.
with strategy.scope():
  gan = DGMR()
  optimizer = Optimizer(
                  gan, num_epochs=num_epochs, gen_batch_size=batch_size, 
                  disc_lr=2E-4, gen_lr=5E-5, disc_beta1=0.0, disc_beta2=0.999, 
                  gen_beta1=0.0, gen_beta2=0.999
              )

@tf.function
def train_step(iterator):

  def get_data_batch(train_set_batched):
    """Returns data batch.

    This function should return a pair of (input sequence, target unroll sequence)
    of image frames for a given batch size, with the following dimensions:
    batch_inputs are of size [batch_size, 4, 256, 256, 1],
    batch_targets are of size [batch_size, 18, 256, 256, 1].

    Args:
      batch_size: The batch size, int.

    Returns:
      batch_inputs:
      batch_targets: Data for training.
    """
    
    batch_inputs = train_set_batched['radar_frames'][:,0:4,:,:,:]
    batch_targets = train_set_batched['radar_frames'][:,4:22,:,:,:]

    return batch_inputs, batch_targets

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

  def disc_step(batch_inputs, batch_targets, lr_mult=1.):
    """Updates the discriminator once on the given batch of (images, labels)."""
    gan = optimizer.gan
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

    # Aggregate the gradients from the full batch.
    replica_ctx_disc = tf.distribute.get_replica_context()
    disc_grads = replica_ctx_disc.all_reduce(tf.distribute.ReduceOp.MEAN, disc_grads)

    if optimizer.decay_disc_lr:
      optimizer.disc_lr.assign(optimizer.init_disc_lr * lr_mult)
    optimizer.disc_opt.apply(disc_grads, disc_params)
    #print("COMPLETED DISC STEP")
    return loss

  def gen_step(batch_inputs, batch_targets, lr_mult=1.):
    """Updates the generator once."""
    gan = optimizer.gan
    num_samples_per_input = 3
    with tf.GradientTape() as tape:
      gen_samples = [ gan.generate(batch_inputs) for _ in range(num_samples_per_input)]
      grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),batch_targets)
      gen_sequences = [tf.concat([batch_inputs, x], axis=1) for x in gen_samples]
      real_sequence = tf.concat([batch_inputs, batch_targets], axis=1)
      
      generated_scores = []
      for g_seq in gen_sequences:
        concat_inputs = tf.concat([real_sequence, g_seq], axis=0)
        concat_outputs = gan.discriminate(concat_inputs)
        score_real, score_generated = tf.split(concat_outputs, 2, axis=0)
        generated_scores.append(score_generated)

      gen_disc_loss = loss_hinge_gen(tf.concat(generated_scores, axis=0))
      loss = gen_disc_loss + 20.0 * grid_cell_reg
    
    gen_params = gan.generator.trainable_variables
    gen_grads = tape.gradient(loss, gen_params)

    # Aggregate the gradients from the full batch.
    replica_ctx_gen = tf.distribute.get_replica_context()
    gen_grads = replica_ctx_gen.all_reduce(tf.distribute.ReduceOp.MEAN, gen_grads)

    if optimizer.decay_gen_lr:
      optimizer.gen_lr.assign(optimizer.init_gen_lr * lr_mult)
    optimizer.gen_opt.apply(gen_grads, gen_params)
    #print("COMPLETED GEN STEP")
    return loss

  def _get_lr_mult(epoch=1):
    # Linear decay to 0.
    decay_epoch = tf.cast(epoch - optimizer.decay_lr_start_epoch, tf.float32)
    if decay_epoch < tf.constant(0, dtype=tf.float32):
      return tf.constant(1., dtype=tf.float32)
    num_decay_epochs = tf.cast(optimizer.num_epochs - optimizer.decay_lr_start_epoch,
                              dtype=tf.float32)
    return (num_decay_epochs - decay_epoch) / num_decay_epochs

  def step(train_batches_1, train_batches_2, epoch=1):

    batch_inputs_1, batch_targets_1 = get_data_batch(train_batches_1)
    batch_inputs_2, batch_targets_2 = get_data_batch(train_batches_2)
    lr_mult = _get_lr_mult(epoch)

    disc_step(batch_inputs_1, batch_targets_1, lr_mult=lr_mult)
    disc_step(batch_inputs_2, batch_targets_2, lr_mult=lr_mult)
    l =  gen_step(batch_inputs_2,batch_targets_2,lr_mult=lr_mult)
    return l

  per_replica_loss = strategy.run(step, args=(next(iterator),next(iterator)))
  return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)


train_iterator = iter(train_dataset)
num_gen_steps = 500000

steps_with_progress = tqdm.tqdm(range(num_gen_steps*num_epochs),
                                unit='radars', unit_scale=batch_size,
                                position=0)

checkpoint_path = "gs://now_casting_tpu_eu/trainer_v1/checkpoints/train"
ckpt = tf.train.Checkpoint(generator=gan.generator,
                           discriminator=gan.discriminator,
                           generator_optimizer=optimizer.gen_opt,
                           discriminator_optimizer=optimizer.disc_opt)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

total_loss = 0.0
num_batches = 0
for step_num in steps_with_progress:
  total_loss += train_step(train_iterator)
  num_batches+=1
  if step_num%10==0:
    print(f'LOSS: {total_loss/num_batches}, STEP: {step_num}')
  if step_num and (step_num % 1000 == 0):
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for step {} at {}'.format(step_num,
                                                         ckpt_save_path))
    tqdm.tqdm.write(
        '\nStep = {}/{} (lr_mult = {:0.02f}, loss = {}) done.'.format(
            step_num, num_gen_steps, 1, total_loss/num_batches))

print('Step = {}/{} (lr_mult = {:0.02f}, loss = {}) done.'.format(
    num_epochs, num_gen_steps, 1, total_loss/num_batches))

ckpt_save_path = ckpt_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(num_gen_steps,
                                                         ckpt_save_path))




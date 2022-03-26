from .optimizer import Optimizer
from .dgmr import DGMR
import tensorflow.compat.v1 as tf
import os
import tqdm

#-----------------------------------------------------------------------------------
#FOR TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))
#------------------------------------------------------------------------------------


EXAMPLE_DATASET_BUCKET_PATH = "gs://dm-nowcasting/datasets/nowcasting_open_source_osgb/nimrod_osgb_1000m_yearly_splits/radar/20200718"
DATASET_ROOT_DIR = EXAMPLE_DATASET_BUCKET_PATH

'''
_FEATURES = {name: tf.io.FixedLenFeature(shape, dtype)
             for name, shape, dtype in [
               ("radar", (20,1000,1500), tf.float32), ("grid_res",(), tf.float32),
               ("x1_lowerleft",(), tf.float32), ("y1_lowerleft",(), tf.float32),
               ("x2_upperright",(), tf.float32), ("y2_upperright",(), tf.float32),
               ("end_time_timestamp",(20,), tf.int64),
             ]}

_SHAPE_BY_SPLIT_VARIANT = {
    ("train"): (20,1000, 1500,1),
}

_MM_PER_HOUR_INCREMENT = 1/32.
_MAX_MM_PER_HOUR = 128.
_MASK_VALUE = np.nan

def parse_and_preprocess_row(row, split, variant):
  result = tf.io.parse_example(row, _FEATURES)
  shape = _SHAPE_BY_SPLIT_VARIANT[(split)]
  radar_bytes = result.pop("radar")
  radar_float32 = tf.reshape(radar_bytes,shape)[:,0:256,0:256,:]
  mask = tf.not_equal(radar_float32, _MASK_VALUE)
  radar = tf.cast(radar_float32, tf.float32)
  radar = tf.clip_by_value(
      radar, -1, _MAX_MM_PER_HOUR)
  result["radar_frames"] = radar
  result["radar_mask"] = mask
  return result

def reader(split="train", variant=None, shuffle_files=False):

  #shards_glob = os.path.join(DATASET_ROOT_DIR, split, variant, "*.tfrecord.gz")
  shards_glob = os.path.join(DATASET_BUCKET_PATH,"*.tfrecord*.gz")
  shard_paths = tf.io.gfile.glob(shards_glob)
  shards_dataset = tf.data.Dataset.from_tensor_slices(shard_paths)
  if shuffle_files:
    shards_dataset = shards_dataset.shuffle(buffer_size=len(shard_paths))
  return (
      shards_dataset
      .interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
                  num_parallel_calls=tf.data.AUTOTUNE,
                  deterministic=not shuffle_files)
      .map(lambda row: parse_and_preprocess_row(row, split, variant),
           num_parallel_calls=tf.data.AUTOTUNE)
      # Do your own subsequent repeat, shuffle, batch, prefetch etc as required.
  )

'''

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
_BATCH_SIZE = 16


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


def reader(split="train", variant="random_crops_256", shuffle_files=False):
  """Reader for open-source nowcasting datasets.
  
  Args:
    split: Which yearly split of the dataset to use:
      "train": Data from 2016 - 2018, excluding the first day of each month.
      "valid": Data from 2016 - 2018, only the first day of the month.
      "test": Data from 2019.
    variant: Which variant to use. The available variants depend on the split:
      "random_crops_256": Available for the training split. 24x256x256 pixel
        crops, sampled with a bias towards crops containing rainfall. Crops at
        all spatial and temporal offsets were able to be sampled, some crops may
        overlap.
      "subsampled_tiles_256_20min_stride": Available for the validation set.
        Non-spatially-overlapping 24x256x256 pixel crops, subsampled from a
        regular spatial grid with stride 256x256 pixels, and a temporal stride
        of 20mins (4 timesteps at 5 minute resolution). Sampling favours crops
        containing rainfall.
      "subsampled_overlapping_padded_tiles_512_20min_stride": Available for the
        test set. Overlapping 24x512x512 pixel crops, subsampled from a
        regular spatial grid with stride 64x64 pixels, and a temporal stride
        of 20mins (4 timesteps at 5 minute resolution). Subsampling favours
        crops containing rainfall.
        These crops include extra spatial context for a fairer evaluation of
        the PySTEPS baseline, which benefits from this extra context. Our other
        models only use the central 256x256 pixels of these crops.
      "full_frame_20min_stride": Available for the test set. Includes full
        frames at 24x1536x1280 pixels, every 20 minutes with no additional
        subsampling.
    shuffle_files: Whether to shuffle the shard files of the dataset
      non-deterministically before interleaving them. Recommended for the
      training set to improve mixing and read performance (since
      non-deterministic parallel interleave is then enabled).

  Returns:
    A tf.data.Dataset whose rows are dicts with the following keys:

    "radar_frames": Shape TxHxWx1, float32. Radar-based estimates of
      ground-level precipitation, in units of mm/hr. Pixels which are masked
      will take on a value of -1/32 and should be excluded from use as
      evaluation targets. The coordinate reference system used is OSGB36, with
      a spatial resolution of 1000 OSGB36 coordinate units (approximately equal
      to 1km). The temporal resolution is 5 minutes.
    "radar_mask": Shape TxHxWx1, bool. A binary mask which is False
      for pixels that are unobserved / unable to be inferred from radar
      measurements (e.g. due to being too far from a radar site). This mask
      is usually static over time, but occasionally a whole radar site will
      drop in or out resulting in large changes to the mask, and more localised
      changes can happen too. 
    "sample_prob": Scalar float. The probability with which the row was
      sampled from the overall pool available for sampling, as described above
      under 'variants'. We use importance weights proportional to 1/sample_prob
      when computing metrics on the validation and test set, to reduce bias due
      to the subsampling.
    "end_time_timestamp": Scalar int64. A timestamp for the final frame in
      the example, in seconds since the UNIX epoch (1970-01-01 00:00:00 UTC).
    "osgb_extent_left", "osgb_extent_right", "osgb_extent_top",
    "osgb_extent_bottom":
      Scalar int64s. Spatial extent for the crop in the OSGB36 coordinate
      reference system.
  """
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
      .batch(_BATCH_SIZE)
      #.prefetch(8)

  )

def get_data_batch():
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
  train_set_batched = next(train_dataset)
  batch_inputs = train_set_batched['radar_frames'][:,0:4,:,:,:]
  batch_targets = train_set_batched['radar_frames'][:,4:22,:,:,:]

  return batch_inputs, batch_targets


dataset = reader(split="train", variant="random_crops_256")
                                             
num_epochs = 5
num_disc_steps = 2
batch_size = _BATCH_SIZE

# We'll turn the step function which updates our models into a tf.function using
# autograph. This makes training much faster. If debugging, you can turn this
# off by setting `debug = True`.
debug = True
gan = DGMR()
optimizer = Optimizer(
                gan, num_epochs=num_epochs, gen_batch_size=batch_size, 
                disc_lr=2E-4, gen_lr=5E-5, disc_beta1=0.0, disc_beta2=0.999, 
                gen_beta1=0.0, gen_beta2=0.999
            )

step = optimizer.step
if not debug:
  step = tf.function(step)

'''
train_dataset = iter(dataset)
num_examples = 15000000
total_batch_size_per_step = batch_size * num_disc_steps
steps_per_epoch = num_examples // total_batch_size_per_step
'''
train_dataset = iter(dataset)
num_gen_steps = 1000000

steps_with_progress = tqdm.tqdm(range(num_gen_steps*num_disc_steps),
                                unit='radars', unit_scale=batch_size,
                                position=0)


checkpoint_path = ""
ckpt = tf.train.Checkpoint(generator=gan.generator,
                           discriminator=gan.discriminator,
                           generator_optimizer=optimizer.gen_opt,
                           discriminator_optimizer=optimizer.disc_opt)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

for step_num in steps_with_progress:
  #epoch = tf.constant(int(step_num / steps_per_epoch)) 
  train_batches = [get_data_batch() for _ in range(num_disc_steps)]
  loss, lr_mult = step(train_batches, epoch=1)
  if step_num%10==0:
    print(f'LOSS: {loss}, STEP: {step_num}')
  if step_num and (step_num % 1000 == 0):
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for step {} at {}'.format(step_num,
                                                         ckpt_save_path))
    tqdm.tqdm.write(
        '\nStep = {}/{} (lr_mult = {:0.02f}, loss = {}) done.'.format(
            step_num, num_gen_steps, lr_mult.numpy(), loss.numpy()))

print('Step = {}/{} (lr_mult = {:0.02f}, loss = {}) done.'.format(
    num_epochs, num_gen_steps, lr_mult.numpy(), loss.numpy()))

ckpt_save_path = ckpt_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(num_gen_steps,
                                                         ckpt_save_path))


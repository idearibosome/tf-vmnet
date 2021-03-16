import argparse
import importlib
import os
import time

import numpy as np
import tensorflow as tf

import dataloaders
import models

FLAGS = tf.flags.FLAGS

DEFAULT_DATALOADER = 'basic_loader'
DEFAULT_MODEL = 'vmnet'

if __name__ == '__main__':
  tf.flags.DEFINE_string('dataloader', DEFAULT_DATALOADER, 'Name of the data loader.')
  tf.flags.DEFINE_string('model', DEFAULT_MODEL, 'Name of the model.')
  tf.flags.DEFINE_string('scales', '2,3,4', 'Scales of the input images. Use the \',\' character to specify multiple scales (e.g., 2,3,4).')
  tf.flags.DEFINE_string('cuda_device', '-1', 'CUDA device index to be used in the validation. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify this to employ GPUs.')

  tf.flags.DEFINE_string('restore_path', None, 'Checkpoint path to be restored. Specify this to resume the training or use pre-trained parameters.')
  tf.flags.DEFINE_string('restore_target', None, 'Target of the restoration.')
  tf.flags.DEFINE_integer('restore_global_step', 0, 'Global step of the restored model. Some models may require to specify this.')

  tf.flags.DEFINE_string('save_path', None, 'Base path of the upscaled images. Specify this to save the upscaled images.')

  tf.flags.DEFINE_integer('shave_size', 4, 'Amount of pixels to crop the borders of the images before calculating quality metrics.')
  tf.flags.DEFINE_boolean('ensemble_only', False, 'Calculate (and save) ensembled image only.')

  # parse data loader and model first and import them
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--dataloader', default=DEFAULT_DATALOADER)
  pre_parser.add_argument('--model', default=DEFAULT_MODEL)
  pre_parsed = pre_parser.parse_known_args()[0]
  if (pre_parsed.dataloader is not None):
    DATALOADER_MODULE = importlib.import_module('dataloaders.' + pre_parsed.dataloader)
  if (pre_parsed.model is not None):
    MODEL_MODULE = importlib.import_module('models.' + pre_parsed.model)


def _clip_image(image):
  return np.clip(np.round(image), a_min=0, a_max=255)

def _shave_image(image, shave_size=4):
  return image[shave_size:-shave_size, shave_size:-shave_size]

def _fit_truth_image_size(output_image, truth_image):
  return truth_image[0:output_image.shape[0], 0:output_image.shape[1]]

def _image_psnr(output_image, truth_image):
  diff = truth_image - output_image
  mse = np.mean(np.power(diff, 2))
  psnr = 10.0 * np.log10(255.0 ** 2 / mse)
  return psnr

def _image_rmse(output_image, truth_image):
  diff = truth_image - output_image
  rmse = np.sqrt(np.mean(diff ** 2))
  return rmse


def main(unused_argv):
  # initialize
  FLAGS.vmnet_intermediate_outputs = True
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
  scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))
  tf.logging.set_verbosity(tf.logging.INFO)

  # data loader
  dataloader = DATALOADER_MODULE.create_loader()
  dataloader.prepare()

  # image saving session
  if (FLAGS.save_path is not None):
    tf_image_save_graph = tf.Graph()
    with tf_image_save_graph.as_default():
      tf_image_save_path = tf.placeholder(tf.string, [])
      tf_image_save_image = tf.placeholder(tf.float32, [None, None, 3])
      
      tf_image = tf_image_save_image
      tf_image = tf.round(tf_image)
      tf_image = tf.clip_by_value(tf_image, 0, 255)
      tf_image = tf.cast(tf_image, tf.uint8)
      
      tf_image_png = tf.image.encode_png(tf_image)
      tf_image_save_op = tf.write_file(tf_image_save_path, tf_image_png)

      tf_image_init = tf.global_variables_initializer()
      tf_image_session = tf.Session(config=tf.ConfigProto(
          device_count={'GPU': 0}
      ))
      tf_image_session.run(tf_image_init)

  # model
  model = MODEL_MODULE.create_model()
  model.prepare(is_training=False, global_step=FLAGS.restore_global_step)

  # model > restore
  model.restore(ckpt_path=FLAGS.restore_path, target=FLAGS.restore_target)
  tf.logging.info('restored the model')

  # validate
  num_total_outputs = FLAGS.vmnet_recursions // FLAGS.vmnet_recursion_frequency
  modules_average_psnr_dict = {}
  modules_average_rmse_dict = {}
  for scale in scale_list:
    modules_average_psnr_dict[scale] = []
    modules_average_rmse_dict[scale] = []

  num_images = dataloader.get_num_images()
  for scale in scale_list:
    psnr_list = []
    rmse_list = []
    for i in range(num_total_outputs+1):
      psnr_list.append([])
      rmse_list.append([])

    for image_index in range(num_images):
      input_image, truth_image, image_name = dataloader.get_image_pair(image_index=image_index, scale=scale)

      if (image_index == 0):
        _ = model.upscale(input_list=[input_image], scale=scale)

      output_images = model.upscale(input_list=[input_image], scale=scale)

      output_image_ensemble = np.zeros_like(output_images[0][0])
      ensemble_factor_total = 0.0
      
      for i in range(num_total_outputs):
        num_recursions = (i + 1) * FLAGS.vmnet_recursion_frequency

        output_image = output_images[i][0]

        ensemble_factor = 1.0 / (2.0 ** (num_total_outputs-num_recursions))
        output_image_ensemble = output_image_ensemble + (output_image * ensemble_factor)
        ensemble_factor_total += ensemble_factor

        if (not FLAGS.ensemble_only):
          if (FLAGS.save_path is not None):
            output_image_path = os.path.join(FLAGS.save_path, 't%d' % (num_recursions), 'x%d' % (scale), os.path.splitext(image_name)[0]+'.png')
            tf_image_session.run(tf_image_save_op, feed_dict={tf_image_save_path:output_image_path, tf_image_save_image:output_image})

          truth_image = _clip_image(truth_image)
          output_image = _clip_image(output_image)

          truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

          truth_image_shaved = _shave_image(truth_image, shave_size=FLAGS.shave_size)
          output_image_shaved = _shave_image(output_image, shave_size=FLAGS.shave_size)

          psnr = _image_psnr(output_image=output_image_shaved, truth_image=truth_image_shaved)
          rmse = _image_rmse(output_image=output_image_shaved, truth_image=truth_image_shaved)

          tf.logging.info('t%d, x%d, %d/%d, psnr=%.2f, rmse=%.2f' % (num_recursions, scale, image_index+1, num_images, psnr, rmse))

          psnr_list[i].append(psnr)
          rmse_list[i].append(rmse)
      
      output_image = output_image_ensemble / ensemble_factor_total

      if (FLAGS.save_path is not None):
        output_image_path = os.path.join(FLAGS.save_path, 'ensemble', 'x%d' % (scale), os.path.splitext(image_name)[0]+'.png')
        tf_image_session.run(tf_image_save_op, feed_dict={tf_image_save_path:output_image_path, tf_image_save_image:output_image})

      truth_image = _clip_image(truth_image)
      output_image = _clip_image(output_image)

      truth_image = _fit_truth_image_size(output_image=output_image, truth_image=truth_image)

      truth_image_shaved = _shave_image(truth_image, shave_size=FLAGS.shave_size)
      output_image_shaved = _shave_image(output_image, shave_size=FLAGS.shave_size)

      psnr = _image_psnr(output_image=output_image_shaved, truth_image=truth_image_shaved)
      rmse = _image_rmse(output_image=output_image_shaved, truth_image=truth_image_shaved)

      tf.logging.info('ensemble, x%d, %d/%d, psnr=%.2f, rmse=%.2f' % (scale, image_index+1, num_images, psnr, rmse))

      psnr_list[num_total_outputs].append(psnr)
      rmse_list[num_total_outputs].append(rmse)

    for i in range(num_total_outputs+1):
      average_psnr = np.mean(psnr_list[i])
      modules_average_psnr_dict[scale].append(average_psnr)
      average_rmse = np.mean(rmse_list[i])
      modules_average_rmse_dict[scale].append(average_rmse)


  # finalize
  tf.logging.info('finished')
  for scale in scale_list:
    print('- x%d, PSNR and RMSE:' % (scale))
    print(','.join([('%.3f' % x) for x in modules_average_psnr_dict[scale]]))
    print('')
    print(','.join([('%.3f' % x) for x in modules_average_rmse_dict[scale]]))


if __name__ == '__main__':
  tf.app.run()
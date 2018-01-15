import os
import util as ut
from data import Data
import config
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__=="__main__":
    dicom_root = FLAGS.dicom_root
    record_dir = FLAGS.record_dir
    block_shape = [FLAGS.block_shape_1,FLAGS.block_shape_2,FLAGS.block_shape_3]
    origin_meta = ut.organizie_keys(dicom_root)
    data_meta = ut.split_metas(origin_meta)
    data = Data(data_meta,record_dir)
    data.process_data_single(block_shape)
    print "\ndone\n"

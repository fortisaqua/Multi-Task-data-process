import os
import util as ut
from data import Data
import config
import tensorflow as tf
import gc

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__=="__main__":
    dicom_root = FLAGS.dicom_root
    dicom_root_test = FLAGS.dicom_test_root
    record_dir = FLAGS.record_dir
    record_dir_test = FLAGS.record_test_dir
    block_shape = [FLAGS.block_shape_1,FLAGS.block_shape_2,FLAGS.block_shape_3]

    print '''=========================================
    # training data
    # =========================================
    # '''
    origin_meta = ut.organizie_keys(dicom_root)
    data_meta = ut.split_metas(origin_meta)
    data = Data(data_meta,record_dir)
    data.process_data_single(block_shape,'train')
    del data
    gc.collect()

    print '''=========================================
        testing data
        =========================================
        '''
    origin_meta_test = ut.organizie_keys(dicom_root_test)
    data_meta_test = ut.split_metas(origin_meta_test)
    data_test = Data(data_meta_test,record_dir_test)
    data_test.process_data_single(block_shape,'test')
    del data_test
    gc.collect()
    print "\ndone\n"

import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dicom_root',"./multi_task_data_train/",
    'Directory where original dicom files for training stored'
)

tf.app.flags.DEFINE_string(
    'dicom_test_root',"./multi_task_data_test/",
    'Directory where original dicom files for testing stored'
)

tf.app.flags.DEFINE_string(
    'record_dir',"./records_64/",
    'Directory where tfrecord files will be stored'
)

tf.app.flags.DEFINE_string(
    'record_test_dir','./records_test_64/',
    'Directory where test data stored'
)

tf.app.flags.DEFINE_integer(
    'block_shape_1',64,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'step_1',64,
    'step length of number 0 dimension'
)

tf.app.flags.DEFINE_integer(
    'block_shape_2',64,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'step_2',64,
    'step length of number 0 dimension'
)

tf.app.flags.DEFINE_integer(
    'block_shape_3',64,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'step_3',32,
    'step length of number 0 dimension'
)

tf.app.flags.DEFINE_integer(
    'batch_size_train',2,
    'batch size for training'
)

tf.app.flags.DEFINE_integer(
    'batch_size_test',4,
    'batch size for testing'
)

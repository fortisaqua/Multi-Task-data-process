import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dicom_root',"/opt/multi_task_data/",
    'Directory where original dicom files stored'
)

tf.app.flags.DEFINE_string(
    'record_dir',"./records/",
    'Directory where tfrecord files will be stored'
)

tf.app.flags.DEFINE_integer(
    'block_shape_1',64,
    'shape of single data block'
)
tf.app.flags.DEFINE_integer(
    'block_shape_2',64,
    'shape of single data block'
)
tf.app.flags.DEFINE_integer(
    'block_shape_3',64,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'batch_size_train',2,
    'batch size for training'
)

tf.app.flags.DEFINE_integer(
    'batch_size_test',4,
    'batch size for testing'
)
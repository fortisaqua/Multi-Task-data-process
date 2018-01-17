import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dicom_root',"/opt/multi_task_data_train/",
    'Directory where original dicom files for training stored'
)

tf.app.flags.DEFINE_string(
    'dicom_test_root',"/opt/multi_task_data_test/",
    'Directory where original dicom files for testing stored'
)

tf.app.flags.DEFINE_string(
    'record_dir',"./records/",
    'Directory where tfrecord files will be stored'
)

tf.app.flags.DEFINE_string(
    'record_test_dir','/opt/Multi-Task-data-process/records_test/',
    'Directory where test data stored'
)

tf.app.flags.DEFINE_integer(
    'block_shape_1',48,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'step_1',12,
    'step length of number 0 dimension'
)

tf.app.flags.DEFINE_integer(
    'block_shape_2',48,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'step_2',12,
    'step length of number 0 dimension'
)

tf.app.flags.DEFINE_integer(
    'block_shape_3',48,
    'shape of single data block'
)

tf.app.flags.DEFINE_integer(
    'step_3',12,
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
import os
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import glob
import numpy as np
import util as ut
from data import TF_Records
import config
import SimpleITK as ST

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


record_dir = FLAGS.record_test_dir
block_shape = [FLAGS.block_shape_1,FLAGS.block_shape_2,FLAGS.block_shape_3]
batch_size = FLAGS.batch_size_train

with tf.Graph().as_default():
    records = ut.get_records(record_dir)
    records_processor = TF_Records(records,block_shape)
    single_blocks = records_processor.read_records()
    # airway_block = single_blocks['airway']
    # artery_block = single_blocks['artery']
    # lung_block = single_blocks['lung']
    # original_block = single_blocks['original']
    # # using a queue to input data
    '''
    ret['block_loc'] = tf.reshape(tf.decode_raw(features['block_loc'],tf.int16),[6,1])
        ret['project_name'] = features['project_name']
    '''
    queue = tf.RandomShuffleQueue(capacity=8,min_after_dequeue=4,
                                  dtypes=(
                                      single_blocks['airway'].dtype,
                                      single_blocks['artery'].dtype,
                                      single_blocks['lung'].dtype,
                                      single_blocks['original'].dtype,
                                      single_blocks['block_loc'].dtype,
                                      single_blocks['project_name'].dtype
                                      # single_blocks['back_ground'].dtype
                                  ))
    enqueue_op = queue.enqueue((single_blocks['airway'],
                                single_blocks['artery'],
                                single_blocks['lung'],
                                single_blocks['original'],
                                single_blocks['block_loc'],
                                single_blocks['project_name']
                                # single_blocks['back_ground']
                                ))
    (airway_block,artery_block,lung_block,original_block,block_loc,pro_name) = queue.dequeue()
    qr = tf.train.QueueRunner(queue,[enqueue_op]*4)

    # start read samples from records
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess,coord=coord,start=True)

    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    try:
        error_count =0
        output_count = 0
        for i in range(50000):
            # organize a batch of data for training
            airway_np = np.zeros([2,block_shape[0],block_shape[1],block_shape[2]],np.int16)
            artery_np = np.zeros([2,block_shape[0],block_shape[1],block_shape[2]],np.int16)
            lung_np = np.zeros([2,block_shape[0],block_shape[1],block_shape[2]],np.int16)
            original_np = np.zeros([2,block_shape[0],block_shape[1],block_shape[2]],np.int16)

            # store values into data block
            for m in range(2):
                airway_data,artery_data,lung_data,original_data,block_location,project_name = \
                sess.run([airway_block,artery_block,lung_block,original_block,block_loc,pro_name])
                airway_np[m,:,:,:]+=airway_data
                artery_np[m,:,:,:]+=artery_data
                lung_np[m,:,:,:]+=lung_data
                original_np[m,:,:,:]+=original_data
                if output_count<10 and np.sum(np.float32(artery_data))/(block_shape[0]*block_shape[1]*block_shape[2])>0.05:
                    artery_part = ST.GetImageFromArray(artery_data)
                    original_part = ST.GetImageFromArray(original_data)
                    output_root_temp = './output/'+str(output_count)
                    if not os.path.exists(output_root_temp):
                        os.makedirs(output_root_temp)
                    ST.WriteImage(artery_part,output_root_temp+'/artery.vtk')
                    ST.WriteImage(original_part,output_root_temp+'/original.vtk')
                    output_count+=1
                    print output_count
                else:
                    exit()
            if np.max(airway_np)>1 or np.max(artery_np)>1 or np.max(lung_np)>1:
                error_count+=1
            if i % 5 == 0:
                print("%d data shape :%s "%(i,str(np.shape(original_np))))
                airway_sum = np.sum(np.float32(airway_np))
                artery_sum = np.sum(np.float32(artery_np))
                lung_sum = np.sum(np.float32(lung_np))
                # background_sum = np.sum(np.float32(background_np))
                print("masks from %s: at %s \nairway  shape=%s sum=%s percentage=%s\nartery shape=%s sum=%s percentage=%s\nlung shape=%s sum=%s percentage=%s\n"%(
                      str(project_name),
                      str(block_location),
                      str(np.shape(airway_np)),
                      str(airway_sum),
                      str(np.float32(np.sum(np.float32(airway_np))/(2*block_shape[0]*block_shape[1]*block_shape[2]))),
                      str(np.shape(artery_np)),
                      str(artery_sum),
                      str(np.float32(np.sum(np.float32(artery_np))/(2*block_shape[0]*block_shape[1]*block_shape[2]))),
                      str(np.shape(lung_np)),
                      str(lung_sum),
                      str(np.float32(np.sum(np.float32(lung_np))/(2*block_shape[0]*block_shape[1]*block_shape[2]))),
                      ))
        print "%d blocks have problem" % (error_count)
    except Exception,e:
        print e
        # exit(2)
        coord.request_stop(e)
    coord.request_stop()
    coord.join(enqueue_threads)
sess.close()

import tensorflow as tf
import xlrd
import USPS_forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
TEST_NUM = 600
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="USPS_model"
FILE_NAME="USPS_Classification.xlsx"

def backward(data, label):

    x = tf.placeholder(tf.float32, shape = (None, USPS_forward.INPUT_NODE))
    y_ = tf.placeholder(tf.float32, shape = (None, USPS_forward.OUTPUT_NODE))
    y = USPS_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)	
	
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(data) / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            start = (i*BATCH_SIZE)%TEST_NUM
            end = start+BATCH_SIZE
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: data[start:end], y_: label[start:end]})
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    wb = xlrd.open_workbook(filename=FILE_NAME)
    Train_sheet = wb.sheet_by_name('Train Feature')
    Label_sheet = wb.sheet_by_name('Train Label')
    data = []
    label = []
    temp = [0,0,0,0,0,0,0,0,0,0]
    for i in range(TEST_NUM):
        data.append(Train_sheet.row_values(i))
        temp[int(Label_sheet.cell_value(i, 0)) - 1] = 1
        label.append(temp)
        temp = [0,0,0,0,0,0,0,0,0,0]
    print(data)
    print(label)
    backward(data, label)

if __name__ == '__main__':
    main()



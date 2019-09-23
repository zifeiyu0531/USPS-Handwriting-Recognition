#coding:utf-8

import tensorflow as tf
import xlrd
import xlwt
from xlutils.copy import copy
import numpy as np
import USPS_backward
import USPS_forward

FILE_NAME="USPS_Classification.xlsx"
TEST_NUM = 1260

def restore_model(testArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, USPS_forward.INPUT_NODE])
		y = USPS_forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		variable_averages = tf.train.ExponentialMovingAverage(USPS_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(USPS_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				preValue = sess.run(preValue, feed_dict={x:testArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1


def application():
	wb = xlrd.open_workbook(filename=FILE_NAME)
	Test_sheet = wb.sheet_by_name('Test Feature')
	wbk = xlwt.Workbook()
	Label_sheet = wbk.add_sheet('Test Label')
	data = []
	for i in range(TEST_NUM):
		data.append(Test_sheet.row_values(i))
	for i in range(TEST_NUM):
		print("正在处理第" + str(i) +"个数据")
		preValue = restore_model(data[i:i+1])[0]
		Label_sheet.write(i, 0, str(preValue))
	wbk.save('out_put.xls')

def main():
	application()

if __name__ == '__main__':
	main()		

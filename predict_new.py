import os
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf


try:
    # Fix UTF8 output issues on Windows console.
    # Does nothing if package is not installed
	from win_unicode_console import enable
	enable()
except ImportError:
	pass
logging.getLogger().setLevel(logging.INFO)

def load_trained_params(trained_dir):

	#words_index = json.loads(open(trained_dir + 'words_index.json',encoding='utf-8').read())
	labels = json.loads(open(trained_dir + 'labels.json',encoding='utf-8').read())


	return   labels

def load_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep=',')
	select = ['Descript']

	df = df.dropna(axis=0, how='any', subset=select)
	mysent =  df[select[0]].apply(lambda x: x).tolist()
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'Category' in df.columns:
		select.append('Category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return mysent, y_, df,mysent



def predict_unseen_data():
	trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	test_file = sys.argv[2]

	labels = load_trained_params(trained_dir)
	x_, y_, df,mysent = load_test_data(test_file, labels)
	my = x_

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)


	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './predicted_results_' + timestamp + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)


	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		#the signature of the graph
		signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
		export_path =  './SavedModelB'
        # making a prediction from the file .pb
		meta_graph_def = tf.saved_model.loader.load(
			sess,
			[tf.saved_model.tag_constants.SERVING],
			export_path)
		signature = meta_graph_def.signature_def

		       #getting the tensors
		to_input_x = signature[signature_key].inputs['to_input_x'].name


		to_input_x =  sess.graph.get_tensor_by_name(to_input_x)


		to_predictions = signature[signature_key].outputs['to_predictions'].name
		to_predictions =  sess.graph.get_tensor_by_name(to_predictions)

		with sess.as_default():
			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / 4) for batch in batches]

			def predict_step(x_batch):
				#print(x_batch.tolist())
				print(x_batch)

				#print(real_len(x_batch))


				feed_dict = {
					to_input_x: x_batch,
				}
				predictions = sess.run([to_predictions], feed_dict)
				print('----------------------------------------')

				return predictions

			batches = data_helper.batch_iter(list(x_test), 1, 1, shuffle=False)
			predictions, predict_labels = [], []
			i = 0
			for x_batch in batches:
				if len(x_batch) != 0:
					batch_predictions = predict_step(x_batch)[0]
					for batch_prediction in batch_predictions:
						print('Prediction is :',batch_prediction,labels[batch_prediction],' :: ',np.argmax(y_test[i], axis=0),labels[np.argmax(y_test[i], axis=0)])# here we print the result of the prediction
						i = i+1
						predictions.append(batch_prediction)
						predict_labels.append(labels[batch_prediction])

			print('fffffff')
			# Save the predictions back to file
			df['NEW_PREDICTED'] = predict_labels
			columns = sorted(df.columns, reverse=True)
			df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
				logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

if __name__ == '__main__':
	# python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv
	predict_unseen_data()

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def constant():
	x1 = tf.constant([5])
	x2 = tf.constant([6])

	#result = x1 * x2
	result = tf.multiply(x1,x2)

	#sess = tf.Session()
	#print(sess.run(result))
	#sess.close()
	with tf.Session() as sess:
		output = sess.run(result)
		print(output)


def deep_nn():
	''' unput > weight > hidden layer 1 (activation function) 
	> weights > hidden layer 2 (activation function) > weights > output layer
	
	compayre output to intended output > cost function (cross entropy)
	optimization function (optimizer) > minimize cost (AdamOptimizer...SGD,AdaGrad)

	backpropogation

	feedforward + backprop = epoch
'''
	return

mnist = input_data.read_data_sets('/tmp/data/',one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_nodes_hl3 = 500
n_classes = 10

batch_size = 100

x = tf.placeholder('float', [None,28 * 28])
y = tf.placeholder('float')


def nn_model(data):

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([28 * 28 , n_nodes_hl1])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1 , n_nodes_hl2])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2 , n_nodes_hl3])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_classes])),
						'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

	return output
def train_nn(x):
	prediction = nn_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))

	# learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles of feedforward + backprop
	total_epochs = 10

	with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(total_epochs):
				epoch_loss = 0
				for i in range(int(mnist.train.num_examples/batch_size)):
					epoch_x,epoch_y = mnist.train.next_batch(batch_size)
					i, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
					epoch_loss += c
				print('Epoch', epoch, 'completed out of ', total_epochs, 'loss', epoch_loss)

			correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

			accuracy = tf.reduce_mean(tf.cast(correct,'float'))
			print('Accuracy:', accuracy.eval({x:	mnist.test.images,
											  y:	mnist.test.labels}))


if __name__ == '__main__':
	#constant()
	train_nn(x)

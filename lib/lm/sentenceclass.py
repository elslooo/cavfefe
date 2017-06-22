import tensorflow as tf 
import numpy as np
from lib.lm.language_model import LanguageModel


class SentenceClassifier(LanguageModel):

	def __init__(self, max_length, embedding_size, num_hidden, nr_classes,
                 learning_rate = 0.01):

		self.nr_classes = nr_classes
		super(SentenceClassifier, self).__init__(max_length, embedding_size, num_hidden,
                 learning_rate = 0.01)
		

   	def _create_placeholders(self):
	    # X is a batch of sentences. The sentences should be padded to n_steps.
	    self.x = tf.placeholder("float", [ None, self.max_length,
	                                       self.embedding_size ], name = "x")

	    # This is the sequence length for each sentence in the batch. This does
	    # include the start-of-sentence marker, but does not include the
	    # end-of-sentence marker.
	    self.seqlen = tf.placeholder(tf.int32, [ None ], name = "seqlen")

	    # This is the class prediction distribution
	    self.y = tf.placeholder("float", [ None, self.nr_classes],
	                            name = "y")
	
   	def _create_weights(self):
        # These weights are used to get from the output layer of the first LSTM
        # / input layer of the second LSTM to the output layer of the second
        # LSTM. Note that in our implementation, this does not include the
        # feature embedding size.
	    self.out_W = tf.Variable(tf.random_normal([
	        self.num_hidden,
	        self.nr_classes
	    ]))

	    self.out_b = tf.Variable(tf.random_normal([
	        self.nr_classes
	    ]))

	def _build_model(self):
	    # The model consists on one LSTM layer trained to predict classes
	    # given a particular sentence
	    lstm = tf.contrib.rnn.BasicLSTMCell(self.num_hidden, forget_bias = 1.0)


	    outputs, final_state = tf.nn.dynamic_rnn(lstm, self.x,
	                                       dtype = tf.float32)

	    self.pred = tf.matmul(outputs[:, -1], self.out_W) + self.out_b
	    self.pred_softmax = tf.nn.softmax(self.pred)

	    # This is the relevance loss of the language model.
	    self.cost = \
	    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.pred,
	                                                           labels = self.y))
	    optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
	    self.optimizer = optimizer.minimize(self.cost)

	    # Evaluate model
	    correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
	    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	    """
	    This function trains the language model on a batch of sentences.
	    """
	def train(self, x, y, sequence_lengths, session):
	    session.run(self.optimizer, feed_dict = {
	        self.x: x,
	        self.y: y,
	        self.seqlen: sequence_lengths
	    })

   	def evaluate(self, x, y, sequence_lengths, session):

	    	acc, loss = session.run([ self.accuracy, self.cost ], feed_dict = {
	            self.x: x,
	            self.y: y,
	            self.seqlen: sequence_lengths
	        })

	        return acc, loss

	def generate(self, session):
	        sos = [ 1 ] + [ 0 ] * (self.embedding_size - 1)
	        eos = [ 0, 1 ] + [ 0 ] * (self.embedding_size - 2)

	        sentence = [ sos ]

	        for i in range(self.max_length - 1):
	            input    = sentence[:]

	            while len(input) < self.max_length:
	                input = input + [ eos ]

	            input = np.array([ input ])

	            output = session.run(self.pred_softmax, feed_dict = {
	                self.x: input,
	                self.seqlen: [ len(sentence) ]
	            })[0]

	            word_index = np.argmax(output)

	            sentence.append(output)

	        return sentence
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import sys,time,os
from csv import ImageToMatrix,batch_index


class CNN(object):
    """docstring for CNN"""
    def __init__(self,
                n_class,
                batch_size,
                learning_rate,
                height,
                width,
                keep_prob1,
                keep_prob2,
                keep_prob3,
                training_iter,
                train_file_path,
                lambdaa,
                display_step,
                test_batch_size
                ):
        self.n_class=n_class
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.height=height
        self.width=width
        self.Keep_prob1=keep_prob1
        self.Keep_prob2=keep_prob2
        self.Keep_prob3=keep_prob3
        self.training_iter=training_iter
        self.train_file_path=train_file_path
        self.lambdaa=lambdaa
        self.display_step=display_step
        self.test_batch_size=test_batch_size
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.height*self.width])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)
            self.keep_prob3 = tf.placeholder(tf.float32)
        def init_variable(shape):
            initial=tf.random_uniform(shape,-0.1,0.1)
            return tf.Variable(initial)


        with tf.name_scope('weights'):
            self.weights={
                'conv1':init_variable([5,5,1,8]),
                'conv2':init_variable([5,5,8,16]),
                'conv3':init_variable([5,5,16,32]),
                'linear1':init_variable([8*8*32,512]),
                'linear2':init_variable([512,256]),
                'linear3':init_variable([256,self.n_class])
            }
        with tf.name_scope('biases'):
            self.biases={
                'conv1':init_variable([8]),
                'conv2':init_variable([16]),
                'conv3':init_variable([32]),
                'linear1':init_variable([512]),
                'linear2':init_variable([256]),
                'linear3':init_variable([self.n_class])
            }
    def model(self,inputs):
        def weight_variable(shape): 
            initial = tf.truncated_normal(shape, stddev=0.1) 
            return tf.Variable(initial)
        def bias_variable(shape): 
            initial = tf.constant(0.1, shape=shape) 
            return tf.Variable(initial)
        def AcFun(x):
            # return x
            return tf.nn.relu(x)
        def conv2d(x, W): 
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        def max_pool_2x2(x): 
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.it=inputs
        inputs=tf.reshape(inputs,[-1,self.height,self.width,1])
        # print(inputs)
        with tf.name_scope('conv'):

            conv1 = conv2d(inputs, self.weights['conv1']) + self.biases['conv1']
            conv1 = AcFun(conv1)
            pool1 = max_pool_2x2(conv1)
            self.pool1_shape=tf.shape(pool1)
            # output1 = tf.reshape(pool1, [-1, 32, 32, 1])

            conv2 = conv2d(pool1, self.weights['conv2']) + self.biases['conv2']
            conv2 = AcFun(conv2)
            pool2 = max_pool_2x2(conv2)
            self.pool2_shape=tf.shape(pool2)
            # output2 = tf.reshape(pool2, [-1, 110, self.embedding_dim, 1])

            conv3 = conv2d(pool2, self.weights['conv3']) + self.biases['conv3']
            conv3 = AcFun(conv3)
            pool3 = max_pool_2x2(conv3)
            output3 = tf.reshape(pool3 ,[-1, 8*8*32])
            self.pool3_shape=tf.shape(pool3)

        with tf.name_scope('linear'):
            output4 = tf.matmul(output3, self.weights['linear1']) + self.biases['linear1']
            output4 = AcFun(output4)
            output4= tf.nn.dropout(output4, keep_prob=self.keep_prob1)


            output5 = tf.matmul(output4, self.weights['linear2']) + self.biases['linear2']
            output5 = AcFun(output5)
            output5 = tf.nn.dropout(output5, keep_prob=self.keep_prob2)


            output6 = tf.matmul(output5, self.weights['linear3']) + self.biases['linear3']
            output6 = AcFun(output6)
            output6 = tf.nn.dropout(output6, keep_prob=self.keep_prob3)


        with tf.name_scope('softmax'):
            predict = tf.nn.softmax(output6)
            # self.sshape=tf.shape(self.predict)
        return predict


    #将数据喂给feed_dict
    def get_batch_data(self,x,y,batch_size,keep_prob1,keep_prob2,keep_prob3):
        for i in batch_index(len(y), batch_size, 1):
            # print("index={}".format(index))
            # print('type x[i] is {}'.format(type(x[i])))
            # print('type x[i] is {}'.format(type(y[i])))
            # print('type x[i] is {}'.format(type(keep_prob1)))
            # print('type x[i] is {}'.format(type(keep_prob2)))
            # print('type x[i] is {}'.format(type(keep_prob3)))

            feed_dict = {
                self.x: x[i],
                self.y: y[i],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
                self.keep_prob3: keep_prob3
            }
            yield feed_dict, len(i)



    def run(self):

        prob=self.model(self.x)
        #代价函数cost function
        with tf.name_scope('loss'):
            cost= - tf.reduce_sum(self.y*tf.log(prob))
            reg=0.
            str=['conv1','conv2','conv3','linear1','linear2','linear3']
            for s in str:
                reg+=tf.nn.l2_loss(self.weights[s])+tf.nn.l2_loss(self.biases[s])
            cost+=reg*self.lambdaa

        with tf.name_scope('train'):
            global_step=tf.Variable(0,name="tr_global_step",trainable=False)
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,global_step=global_step)

        with tf.name_scope('predict'):
            testpred=tf.argmax(prob,1)
            testpred1=tf.argmax(self.y,1)
            correct_pred=tf.equal(tf.argmax(prob,1),tf.argmax(self.y,1))
            accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
            correct_num=tf.reduce_sum(tf.cast(correct_pred,tf.int32))


        #从路径中读取数据到tr_x,tr_y
        with tf.name_scope('readData'):
            print ('\n---ReadData-------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
            tr_x,tr_y= ImageToMatrix(
                self.train_file_path
            )
            # print(tr_x)
            # print(tr_y)

        with tf.Session() as sess:
            merged_summary_op=tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('logs/', sess.graph)
            sess.run(tf.global_variables_initializer())
            max_acc=0.
            best_Iter=0
            step=0
            for i in range(self.training_iter):
                #从feed_dict拿训练数据
                for feed, _ in self.get_batch_data(tr_x, tr_y, self.batch_size, self.Keep_prob1, self.Keep_prob2,self.Keep_prob3):
                    
                    #运行，得到数据集的特征值
                    _, step,  loss, acc= sess.run([optimizer, global_step, cost, accuracy], feed_dict=feed)



                    # train_summary_writer.add_summary(summary, step)
                    # summary_writer.add_summary(merged_summary_op_, i)
                    # if i == 0:
                    # print('testpredict\n')
                    # print(sess.run(testpred,feed_dict=feed))
                    # print('testpredict1\n')
                    # print(sess.run(testpred1,feed_dict=feed))
                    # print('testpredict3\n')
                    # print(sess.run([prob],feed_dict=feed))
                    # print('testpredict4\n')
                    # print(sess.run([self.it,self.pool1_shape,self.pool2_shape,self.pool3_shape],feed_dict=feed))

                    print ('Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc))

                #测试集检验
                if i%self.display_step==0:
                    acc,loss,cnt,stepp=0.,0.,0,0
                    for test, num in self.get_batch_data(tr_x, tr_y, self.test_batch_size, self.Keep_prob1, self.Keep_prob2,self.Keep_prob3):
                        _loss, _acc = sess.run([cost, correct_num], feed_dict=test)
                        acc += _acc
                        loss += _loss * num
                        # print("num is {}".format(num))
                        cnt += num
                    loss = loss / cnt #num
                    acc = acc / cnt   #num
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                    
                    # summary = sess.run(summary_test, feed_dict={
                    #                    test_loss: loss, test_acc: acc})
                    # test_summary_writer.add_summary(summary, step)
                    print ('----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))
                    print ('Iter {}: test loss={:.6f}, test acc={:.6f}'.format(step, loss, acc))
                    print ('round {}: max_acc={} BestIter={}\n'.format(i, max_acc, best_Iter))            
            print ('Optimization Finished!')


def main(_):
    #sys.stdout = open('result'+'={}='.format(time.strftime("%Y-%m-%d-%X", time.localtime())).replace(':','-'),'w')
    cnn=CNN(
        n_class=62,
        batch_size=50,
        learning_rate=0.0001,
        height=64,
        width=64,
        keep_prob1=1.0,
        keep_prob2=1.0,
        keep_prob3=1.0,
        training_iter=5000,
        train_file_path='/Users/apple/Desktop/gra_/train/dataset_/',
        lambdaa=0.001,
        display_step=1,
        test_batch_size=50
        )
    cnn.run()



if __name__ == '__main__':
    tf.app.run()










# conv1 64*64 2*2  32*32 8
# conv2 32*32 2*2 16*16 16
# conv3 16*16 2*2 8*8  32


# 8*8*32
# 1*2048
# linear3 1*2048  1*512
# linear4 1*512 1*256
# linear5 1*256 1*62
# softmax 1*62 

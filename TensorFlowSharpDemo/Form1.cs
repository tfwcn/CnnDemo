using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using TensorFlow;

namespace TensorFlowSharpDemo
{
    public partial class Form1 : Form
    {
        TFGraph graph;
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                /*
                using (var session = new TFSession())
                {
                    graph = session.Graph;

                    var a = graph.Const(2);
                    var b = graph.Const(3);
                    //Console.WriteLine("a=2 b=3");

                    // Add two constants
                    var addingResults = session.GetRunner().Run(graph.Add(a, b));
                    var addingResultValue = addingResults.GetValue();
                    //Console.WriteLine("a+b={0}", addingResultValue);
                    label1.Text = String.Format("a+b={0}", addingResultValue);

                    // Multiply two constants
                    var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
                    var multiplyResultValue = multiplyResults.GetValue();
                    //Console.WriteLine("a*b={0}", multiplyResultValue);
                    label1.Text = String.Format("a*b={0}", multiplyResultValue);
                }
                //*/
                using (var session = new TFSession())
                {
                    graph = session.Graph;
                    var x = graph.Placeholder(TFDataType.Float, new TFShape(-1, 784));//定义变量，输入值
                    var y_ = graph.Placeholder(TFDataType.Float, new TFShape(-1, 10));//定义变量，输出值
                    var W = graph.Variable(graph.ParameterizedTruncatedNormal(
                        graph.Const(new TFTensor(new float[784, 10])),
                        graph.Const(0, TFDataType.Float), graph.Const(0.1, TFDataType.Float), graph.Const(-1, TFDataType.Float), graph.Const(1, TFDataType.Float)));//权重
                    var b = graph.Variable(graph.Const(new TFTensor(new float[10])));//偏置

                    var y = graph.Softmax(graph.Add(graph.MatMul(x, W), b));//前向传播公式
                    var cross_entropy = graph.Mul(graph.ReduceSum(graph.Mul(y_, graph.Log(y))), graph.Const(-1));//损失函数，交叉熵
                    //var train_step = graph.ResourceApplyGradientDescent(graph.Const(0.01),,);//#训练步骤，梯度下降法，步长0.01，,最小偏差值为交叉熵
                    var r= session.GetRunner().Run(W);
                    label1.Text = r.GetValue().ToString();
                }
                /*
sess = tf.InteractiveSession()#启动Session，与底层通信
x = tf.placeholder("float", shape=[None, 784])#定义变量，输入值
y_ = tf.placeholder("float", shape=[None, 10])#定义变量，输出值
W = tf.Variable(tf.zeros([784,10]))#权重
b = tf.Variable(tf.zeros([10]))#偏置
sess.run(tf.initialize_all_variables())#初始化全部变量
y = tf.nn.softmax(tf.matmul(x,W) + b)#前向传播公式
cross_entropy = -tf.reduce_sum(y_*tf.log(y))#损失函数，交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#训练步骤，梯度下降法，步长0.01，,最小偏差值为交叉熵
for i in range(1000):#循环1000次
    batch = mnist.train.next_batch(50)#每次50个集合
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})#运行一步
    print(i)#输出次数

#验证
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#比较结果是否相同
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#把bool值转float，求结果平均
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))#输出结果
                 */
            }
            catch (Exception ex)
            {

            }
        }
    }
}

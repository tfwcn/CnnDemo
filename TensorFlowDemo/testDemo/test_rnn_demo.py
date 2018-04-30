import copy
import numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative


def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
# 这一行声明了一个查找表，这个表是一个实数与对应二进制表示的映射。
# 二进制表示将会是我们网路的输入与输出，所以这个查找表将会帮助我们将实数转化为其二进制表示。
int2binary = {}
# 这里设置了二进制数的最大长度。如果一切都调试好了，你可以把它调整为一个非常大的数。
binary_dim = 8

# 这里计算了跟二进制最大长度对应的可以表示的最大十进制数。8位2进制最大255
largest_number = pow(2, binary_dim)
# print("largest_number:",largest_number)
# 这里生成了十进制数转二进制数的查找表，并将其复制到int2binary里面。虽然说这一步不是必需的，但是这样的话理解起来会更方便。
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
    # print("int2binary[%d]:%s" % (i,int2binary[i]))


# input variables
# 这里设置了学习速率。
alpha = 0.1
# 我们要把两个数加起来，所以我们一次要输入两位字符。如此以来，我们的网络就需要两个输入。
input_dim = 2
# 这是隐含层的大小，回来存储“携带位”。
# 需要注意的是，它的大小比原理上所需的要大。
# 自己尝试着调整一下这个值，然后看看它如何影响收敛速率。
# 更高的隐含层维度会使训练变慢还是变快？更多或是更少的迭代次数？
hidden_dim = 16
# 我们只是预测和的值，也就是一个数。如此，我们只需一个输出。
output_dim = 1


# initialize neural network weights
# 这个权值矩阵连接了输入层与隐含层，如此它就有“imput_dim”行以及“hidden_dim”列（假如你不改参数的话就是2×16）。
synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1
# print("synapse_0:",synapse_0.shape)
# 这个权值矩阵连接了隐含层与输出层，如此它就有“hidden_dim”行以及“output_dim”列（假如你不改参数的话就是16×1）。
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1
# 这个权值矩阵连接了前一时刻的隐含层与现在时刻的隐含层。
# 它同样连接了当前时刻的隐含层与下一时刻的隐含层。
# 如此以来，它就有隐含层维度大小（hidden_dim）的行与隐含层维度大小（hidden_dim）的列（假如你没有修改参数就是16×16）。
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1

# 这里存储权值更新。在我们积累了一些权值更新以后，我们再去更新权值。这里先放一放，稍后我们再详细讨论。
# synapse_0_update = np.zeros_like(synapse_0)
# synapse_1_update = np.zeros_like(synapse_1)
# synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):

    # generate a simple addition problem (a + b = c)
    # 这里我们要随机生成一个在范围内的加法问题。
    # 所以我们生成一个在0到最大值一半之间的整数。
    # 如果我们允许网络的表示超过这个范围，那么把两个数加起来就有可能溢出（比如一个很大的数导致我们的位数不能表示）。
    # 所以说，我们只把加法要加的两个数字设定在小于最大值的一半。
    a_int = np.random.randint(largest_number/2)  # int version
    # 我们查找a_int对应的二进制表示，然后把它存进a里面。
    a = int2binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number/2)  # int version
    b = int2binary[b_int]  # binary encoding

    # true answer
    # 我们计算加法的正确结果。
    c_int = a_int + b_int
    # 把正确结果转化为二进制表示。
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    # 初始化一个空的二进制数组，用来存储神经网络的预测值（便于我们最后输出）。你也可以不这样做，但是我觉得这样使事情变得更符合直觉。
    d = np.zeros_like(c)

    # 重置误差值（这是我们使用的一种记录收敛的方式……可以参考之前关于反向传播与梯度下降的文章）
    overallError = 0

    # 这两个list会每个时刻不断的记录layer 2的导数值与layer 1的值。
    layer_2_deltas = list()
    layer_1_values = list()  # 上一层的值
    # 在0时刻是没有之前的隐含层的，所以我们初始化一个全为0的。
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    # 这个循环是遍历二进制数字。
    for position in range(binary_dim):

        # generate input and output
        # X跟图片中的“layer_0”是一样的，X数组中的每个元素包含两个二进制数，其中一个来自a，一个来自b。
        # 它通过position变量从a，b中检索，从最右边往左检索。
        # 所以说，当position等于0时，就检索a最右边的一位和b最右边的一位。
        # 当position等于1时，就向左移一位。
        # X:(1*2)的矩阵
        X = np.array(
            [[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        # 跟上一行检索的方式一样，但是把值替代成了正确的结果（0或者1）。
        # y:(1*1)的矩阵
        y = np.array([[c[binary_dim - position - 1]]]).T
        # print("X:",X)
        # print("y:",y)

        # hidden layer (input ~+ prev_hidden)
        # 这里就是奥妙所在！一定一定一定要保证你理解这一行！！！
        # 为了建立隐含层，我们首先做了两件事。
        # 第一，我们从输入层传播到隐含层（np.dot(X,synapse_0)）。
        # 然后，我们从之前的隐含层传播到现在的隐含层（np.dot(prev_layer_1.synapse_h)）。
        # 在这里，layer_1_values[-1]就是取了最后一个存进去的隐含层，也就是之前的那个隐含层！
        # 然后我们把两个向量加起来！！！！然后再通过sigmoid函数。
        # 那么，我们怎么结合之前的隐含层信息与现在的输入呢？当每个都被变量矩阵传播过以后，我们把信息加起来
        # layer_1:(1*16)的矩阵，(1*2)*(2*16)=(1*16)
        layer_1 = sigmoid(np.dot(X, synapse_0) +
                          np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)
        # 这行看起来很眼熟吧？这跟之前的文章类似，它从隐含层传播到输出层，即输出一个预测值。
        # layer_1:(1*16)的矩阵,synapse_1:(16*1)的矩阵,layer_2:(1*1)的矩阵
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # did we miss?... if so by how much?
        # 计算一下预测误差（预测值与真实值的差）。
        layer_2_error = y - layer_2
        # 这里我们把导数值存起来（上图中的芥末黄），即把每个时刻的导数值都保留着。
        # 上一层的误差值
        layer_2_deltas.append(
            (layer_2_error[0][0])*sigmoid_output_to_derivative(layer_2))
        # 计算误差的绝对值，并把它们加起来，这样我们就得到一个误差的标量（用来衡量传播）。
        # 我们最后会得到所有二进制位的误差的总和。
        overallError += np.abs(layer_2_error[0][0])

        # decode estimate so we can print it out
        # np.round四舍五入，d记录输出值
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        # 将layer_1的值拷贝到另外一个数组里，这样我们就可以下一个时间使用这个值。
        layer_1_values.append(copy.deepcopy(layer_1))

    #隐藏层的误差
    future_layer_1_delta = np.zeros(hidden_dim)

    # 我们已经完成了所有的正向传播，并且已经计算了输出层的导数，并将其存入在一个列表里了。
    # 现在我们需要做的就是反向传播，从最后一个时间点开始，反向一直到第一个。
    for position in range(binary_dim):
        # 像之前那样，检索输入数据。
        X = np.array([[a[position], b[position]]])
        # 从列表中取出当前的隐含层。-1，最后一个元素
        layer_1 = layer_1_values[-position-1]
        # 从列表中取出前一个隐含层。-2，前一个元素
        prev_layer_1 = layer_1_values[-position-2]

        # error at output layer
        # 从列表中取出当前输出层的误差。
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        # 这一行计算了当前隐含层的误差。
        # 通过当前之后一个时间点的误差和当前输出层的误差计算。
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
            layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        # 我们已经有了反向传播中当前时刻的导数值，那么就可以生成权值更新的量了（但是还没真正的更新权值）。
        # 我们会在完成所有的反向传播以后再去真正的更新我们的权值矩阵，这是为什么呢？
        # 因为我们要用权值矩阵去做反向传播。
        # 如此以来，在完成所有反向传播以前，我们不能改变权值矩阵中的值。
        # synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        # synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        # synapse_0_update += X.T.dot(layer_1_delta)
        synapse_1 += np.atleast_2d(layer_1).T.dot(layer_2_delta) * alpha
        synapse_h += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta) * alpha
        synapse_0 += X.T.dot(layer_1_delta) * alpha

        # 记录当前误差，继续向前传播
        future_layer_1_delta = layer_1_delta

    # 现在我们就已经完成了反向传播，得到了权值要更新的量，所以就赶快更新权值吧（别忘了重置update变量）！
    # synapse_0 += synapse_0_update * alpha
    # synapse_1 += synapse_1_update * alpha
    # synapse_h += synapse_h_update * alpha

    # synapse_0_update *= 0
    # synapse_1_update *= 0
    # synapse_h_update *= 0

    # print out progress
    # 这里仅仅是一些输出日志，便于我们观察中间的计算过程与效果。
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x*pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

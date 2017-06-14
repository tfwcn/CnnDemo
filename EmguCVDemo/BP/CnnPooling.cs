using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 卷积核(池化)
    /// </summary>
    public class CnnPooling
    {
        /// <summary>
        /// 共享权重(所有感知野共享)
        /// </summary>
        public double[,] ShareWeight { get; set; }
        /// <summary>
        /// 偏置
        /// </summary>
        public double OutputOffset { get; set; }
        /// <summary>
        /// 卷积核大小（宽）
        /// </summary>
        public int ConvolutionKernelWidth { get; set; }
        /// <summary>
        /// 卷积核大小（高）
        /// </summary>
        public int ConvolutionKernelHeight { get; set; }
        /// <summary>
        /// 感知野大小（宽）
        /// </summary>
        private int receptiveFieldWidth;
        /// <summary>
        /// 感知野大小（高）
        /// </summary>
        private int receptiveFieldHeight;
        /// <summary>
        /// 输入宽度
        /// </summary>
        private int inputWidth;
        /// <summary>
        /// 输入高度
        /// </summary>
        private int inputHeight;
        /// <summary>
        /// 激活函数类型，1:池化(Mean Pooling),2:池化(Max Pooling)
        /// </summary>
        public int ActivationFunctionType { get; set; }
        /// <summary>
        /// 池化类型，1:平均池化(Mean Pooling),2:最大值池化(Max Pooling)
        /// </summary>
        public int PoolingType { get; set; }
        /// <summary>
        /// 原输入值
        /// </summary>
        public double[,] InputValue { get; set; }
        /// <summary>
        /// 原输出值
        /// </summary>
        public double[,] OutputValue { get; set; }
        /// <summary>
        /// 最大值池化时的最大值指针
        /// </summary>
        private int[,] OutputPoolingMax;

        public CnnPooling(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight, int activationFunctionType = 1, int poolingType = 1)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.PoolingType = poolingType;
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Ceiling(inputWidth / (double)receptiveFieldWidth));
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Ceiling(inputHeight / (double)receptiveFieldWidth));
            OutputPoolingMax = new int[ConvolutionKernelWidth, ConvolutionKernelHeight];
            ShareWeight = new double[receptiveFieldWidth, receptiveFieldHeight];
            InitShareWeight();
        }
        /// <summary>
        /// 前向传播,计算卷积结果
        /// </summary>
        public double[,] CalculatedConvolutionResult(double[,] value)
        {
            InputValue = value;
            double[,] result = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    result[i, j] = CalculatedConvolutionPointResult(value, i, j);//卷积+池化
                    result[i, j] += OutputOffset;//偏置
                    //调用激活函数计算结果
                    result[i, j] = ActivationFunction(result[i, j]);
                }
            }
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算感知野结果(卷积+池化)
        /// </summary>
        /// <returns></returns>
        private double CalculatedConvolutionPointResult(double[,] value, int x, int y)
        {
            double result = 0;
            result = Pooling(value, x, y);
            return result;
        }
        /// <summary>
        /// 激活函数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunction(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case 1:
                    //tanh
                    result = ActivationFunctionTanh(value);
                    break;
                case 2:
                    //PReLU
                    result = ActivationFunctionPReLU(value);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数导数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionDerivative(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case 1:
                    //tanh
                    result = ActivationFunctionTanhDerivative(value);
                    break;
                case 2:
                    //PReLU
                    result = ActivationFunctionPReLUDerivative(value);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanh(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            result = Math.Tanh(value);
            return result;
        }
        /// <summary>
        /// 激活函数导数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanhDerivative(double value)
        {
            double result = 0;
            //激活函数导数计算结果
            result = 1 - Math.Pow(Math.Tanh(value), 2);
            return result;
        }
        /// <summary>
        /// 激活函数（PReLU）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionPReLU(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            if (value >= 0)
            {
                result = value;
            }
            else
            {
                result = 0.05 * value;
            }
            return result;
        }
        /// <summary>
        /// 激活函数导数（PReLU）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionPReLUDerivative(double value)
        {
            double result = 0;
            //激活函数导数计算结果
            if (value > 0)
            {
                result = 1;
            }
            else if (value == 0)
            {
                result = 0;
            }
            else
            {
                result = 0.05;
            }
            return result;
        }
        /// <summary>
        /// 池化
        /// </summary>
        /// <returns></returns>
        private double Pooling(double[,] value, int x, int y)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (PoolingType)
            {
                case 1:
                    //平均池化
                    result = MeanPooling(value, x, y);
                    break;
                case 2:
                    //最大值池化
                    result = MaxPooling(value, x, y);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 池化（平均池化）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double MeanPooling(double[,] value, int x, int y)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < receptiveFieldWidth && receptiveFieldWidth * x + i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && receptiveFieldHeight * y + j < value.GetLength(1); j++)
                {
                    result += value[receptiveFieldWidth * x + i, receptiveFieldHeight * y + j] * ShareWeight[i, j];
                }
            }
            //平均值
            result = result / (receptiveFieldWidth * receptiveFieldHeight);
            return result;
        }
        /// <summary>
        /// 池化（最大值池化）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double MaxPooling(double[,] value, int x, int y)
        {
            double result = value[0, 0];
            OutputPoolingMax[x, y] = 0;
            //计算区域内的最大值
            for (int i = 0; i < receptiveFieldWidth && receptiveFieldWidth * x + i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && receptiveFieldHeight * y + j < value.GetLength(1); j++)
                {
                    if (result < value[receptiveFieldWidth * x + i, receptiveFieldHeight * y + j])
                    {
                        result = value[receptiveFieldWidth * x + i, receptiveFieldHeight * y + j];
                        OutputPoolingMax[x, y] = i + j * receptiveFieldHeight;
                    }
                }
            }
            result *= ShareWeight[OutputPoolingMax[x, y] % receptiveFieldWidth, OutputPoolingMax[x, y] / receptiveFieldHeight];
            return result;
        }
        /// <summary>
        /// 初始化共享权重和偏置
        /// </summary>
        private void InitShareWeight()
        {
            Random random = new Random();
            for (int i = 0; i < ShareWeight.GetLength(0); i++)
            {
                for (int j = 0; j < ShareWeight.GetLength(1); j++)
                {
                    ShareWeight[i, j] = GetRandom(random);
                }
            }
        }
        /// <summary>
        /// 获取随机值
        /// </summary>
        private double GetRandom(Random random)
        {
            double result = 0;
            int inputCount = inputWidth * inputHeight;
            int outputCount = ConvolutionKernelWidth * ConvolutionKernelHeight;
            switch (ActivationFunctionType)
            {
                case 2:
                    //PReLU
                    result = (random.NextDouble() * Math.Abs(inputCount - outputCount) + (inputCount > outputCount ? outputCount : inputCount)) * Math.Sqrt(inputCount / 2);
                    break;
                default:
                    result = (random.NextDouble() * Math.Abs(inputCount - outputCount) + (inputCount > outputCount ? outputCount : inputCount)) * Math.Sqrt(inputCount);
                    break;
            }
            return random.NextDouble();
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <returns></returns>
        public double[,] BackPropagation(double[,] output, double learningRate)
        {
            double[,] result = null;//正确输入值
            //每个核的残差
            double[,] delta = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    delta[i, j] = OutputValue[i, j] - output[i, j];
                }
            }
            double[,] outputTmp = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    outputTmp[i, j] = output[i, j] * receptiveFieldWidth * receptiveFieldHeight;
                }
            }
            //out=sum(in*w)+b
            result = CalculatedBackPropagationResult(outputTmp, learningRate);
            return result;
        }
        /// <summary>
        /// 计算反向传播结果
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResult(double[,] output, double learningRate)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值
            //计算输入值残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    //残差
                    double residual = ActivationFunctionDerivative(OutputValue[i, j]) * (OutputValue[i, j] - output[i, j]);
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldWidth + j2 < inputHeight; j2++)
                        {
                            result[i * receptiveFieldWidth + i2, j * receptiveFieldWidth + j2] += ShareWeight[i2, j2] * residual;
                        }
                    }
                }
            }
            //更新权重和偏置
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    //残差
                    double residual = ActivationFunctionDerivative(OutputValue[i, j]) * (OutputValue[i, j] - output[i, j]);
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldWidth + j2 < inputHeight; j2++)
                        {
                            ShareWeight[i2, j2] -= learningRate * InputValue[i * receptiveFieldWidth + i2, j * receptiveFieldWidth + j2] * residual / (ConvolutionKernelWidth * ConvolutionKernelHeight * receptiveFieldWidth * receptiveFieldHeight);
                        }
                    }
                    OutputOffset -= learningRate * residual / (ConvolutionKernelWidth * ConvolutionKernelHeight);
                }
            }
            //计算正确输入值
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] = InputValue[i, j] - result[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（平均池化）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultMeanPooling(double[,] output)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出/感知野大小，放入原来位置
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] = output[i / receptiveFieldWidth, j / receptiveFieldHeight] / (receptiveFieldWidth * receptiveFieldHeight);
                }
            }
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（最大值池化）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultMaxPooling(double[,] output)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出放入原来最大值位置，其余位置设置成0
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    if (OutputPoolingMax[i / receptiveFieldWidth, j / receptiveFieldHeight] == i % receptiveFieldWidth + j % receptiveFieldHeight * receptiveFieldHeight)
                        result[i, j] = output[i / receptiveFieldWidth, j / receptiveFieldHeight];
                    else
                        result[i, j] = 0;
                }
            }
            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 卷积核
    /// </summary>
    public class CnnKernel
    {
        /// <summary>
        /// 共享权重(所有感知野共享)
        /// </summary>
        public double[,] ShareWeight { get; set; }
        /// <summary>
        /// 共享偏置
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
        /// 偏移值（宽）
        /// </summary>
        private int offsetWidth;
        /// <summary>
        /// 偏移值（高）
        /// </summary>
        private int offsetHeight;
        /// <summary>
        /// 激活函数类型，1:tanh,2:池化(Mean Pooling),3:池化(Max Pooling)
        /// </summary>
        public int ActivationFunctionType;
        /// <summary>
        /// 原输入值
        /// </summary>
        public double[,] InputValue { get; set; }
        /// <summary>
        /// 原输出值
        /// </summary>
        public double[,] OutputValue { get; set; }

        public CnnKernel(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight, int offsetWidth, int offsetHeight, int activationFunctionType = 1)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.offsetWidth = offsetWidth;
            this.offsetHeight = offsetHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Ceiling((inputWidth - receptiveFieldWidth) / (double)offsetWidth)) + 1;
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Ceiling((inputHeight - receptiveFieldHeight) / (double)offsetHeight)) + 1;
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
                    result[i, j] = CalculatedConvolutionPointResult(value, i, j);
                }
            }
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算感知野结果
        /// </summary>
        /// <returns></returns>
        private double CalculatedConvolutionPointResult(double[,] value, int x, int y)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < receptiveFieldWidth && offsetWidth * x + i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && offsetHeight * y + j < value.GetLength(1); j++)
                {
                    result += value[offsetWidth * x + i, offsetHeight * y + j] * ShareWeight[i, j];
                }
            }
            //调用激活函数计算结果
            result = ActivationFunction(result + OutputOffset);
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
        /// 初始化共享权重
        /// </summary>
        private void InitShareWeight()
        {
            Random random = new Random();
            for (int i = 0; i < ShareWeight.GetLength(0); i++)
            {
                for (int j = 0; j < ShareWeight.GetLength(1); j++)
                {
                    ShareWeight[i, j] = random.NextDouble();
                }
            }
            OutputOffset = random.NextDouble();
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
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns></returns>
        public double[,] BackPropagation(double[,] output, double learningRate)
        {
            double[,] result = null;//正确输入值
            result = CalculatedBackPropagationResultTanh(output, learningRate);
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（tanh）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultTanh(double[,] output, double learningRate)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值
            //180度反转权限矩阵
            /*double[,] ShareWeight180 = new double[receptiveFieldWidth, receptiveFieldHeight];
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    ShareWeight180[i, j] = ShareWeight[receptiveFieldWidth - i - 1, receptiveFieldHeight - j - 1];
                }
            }*/
            //计算输入值残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    //残差
                    double residual = ActivationFunctionDerivative(OutputValue[i, j]) * (OutputValue[i, j] - output[i, j]);
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * offsetWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * offsetHeight + j2 < inputHeight; j2++)
                        {
                            result[i * offsetWidth + i2, j * offsetHeight + j2] += learningRate * ShareWeight[i2, j2] * residual;
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
                    for (int i2 = 0; i2 < receptiveFieldWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight; j2++)
                        {
                            ShareWeight[i2, j2] -= learningRate * residual / (ConvolutionKernelWidth * ConvolutionKernelHeight);
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
    }
}

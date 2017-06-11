using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 卷积核(池化)
    /// </summary>
    public class CnnPooling : ICnnNode
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
        /// 激活函数类型，1:池化(Mean Pooling),2:池化(Max Pooling)
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
        /// <summary>
        /// 最大值池化时的最大值指针
        /// </summary>
        private int[,] OutputPoolingMax;

        public CnnPooling(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight, int activationFunctionType = 1)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Floor(inputWidth / (double)receptiveFieldWidth));
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Floor(inputHeight / (double)receptiveFieldWidth));
            ShareWeight = new double[receptiveFieldWidth, receptiveFieldHeight];
            InitShareWeight();
        }
        /// <summary>
        /// 前向传播,计算卷积结果
        /// </summary>
        public double[,] CalculatedConvolutionResult(double[,] value)
        {
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
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case 1:
                    //平均池化
                    result = ActivationFunctionMeanPooling(value, x, y);
                    break;
                case 2:
                    //最大值池化
                    result = ActivationFunctionMaxPooling(value, x, y);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数（平均池化）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionMeanPooling(double[,] value, int x, int y)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < receptiveFieldWidth && i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && j < value.GetLength(1); j++)
                {
                    result += value[receptiveFieldWidth * x + i, receptiveFieldHeight * y + j];
                }
            }
            //平均值
            result = result / (receptiveFieldWidth * receptiveFieldHeight);
            return result;
        }
        /// <summary>
        /// 激活函数（最大值池化）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionMaxPooling(double[,] value, int x, int y)
        {
            double result = value[0, 0];
            //计算区域内的最大值
            for (int i = 0; i < receptiveFieldWidth && i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && j < value.GetLength(1); j++)
                {
                    if (result < value[receptiveFieldWidth * x + 1, receptiveFieldHeight * y + j])
                        result = value[receptiveFieldWidth * x + 1, receptiveFieldHeight * y + j];
                }
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
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns></returns>
        public void BackPropagation(double[,] input, double[,] output, double learningRate)
        {
            switch (ActivationFunctionType)
            {
                case 1:
                    //平均池化
                    CalculatedBackPropagationResultMeanPooling(input, output, learningRate);
                    break;
                case 2:
                    //最大值池化
                    CalculatedBackPropagationResultMaxPooling(input, output, learningRate);
                    break;
            }
        }
        /// <summary>
        /// 计算反向传播结果（平均池化）
        /// </summary>
        /// <returns></returns>
        private void CalculatedBackPropagationResultMeanPooling(double[,] input, double[,] output, double learningRate)
        {
            double outputValueSum = 0;//和
            //计算总和
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    outputValueSum = OutputValue[i, j];
                }
            }
            //更新权重
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    ShareWeight[i, j] -= learningRate * (outputValueSum / (ConvolutionKernelWidth * receptiveFieldHeight));
                }
            }
            //调整输出
            CalculatedConvolutionResult(input);
        }
        /// <summary>
        /// 计算反向传播结果（最大值池化）
        /// </summary>
        /// <returns></returns>
        private void CalculatedBackPropagationResultMaxPooling(double[,] input, double[,] output, double learningRate)
        {
            double outputValueMax = OutputValue[0, 0];//最大值
            //计算最大值
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    if (outputValueMax < OutputValue[i, j])
                        outputValueMax = OutputValue[i, j];
                }
            }
            //更新权重
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    ShareWeight[i, j] -= learningRate * outputValueMax;
                }
            }
            //调整输出
            CalculatedConvolutionResult(input);
        }
    }
}

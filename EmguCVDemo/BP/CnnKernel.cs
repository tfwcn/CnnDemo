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

        public CnnKernel(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight, int offsetWidth, int offsetHeight, int activationFunctionType = 1)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.offsetWidth = offsetWidth;
            this.offsetHeight = offsetHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Floor((inputWidth - receptiveFieldWidth) / (double)offsetWidth)) + 1;
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Floor((inputHeight - receptiveFieldHeight) / (double)offsetHeight)) + 1;
            ShareWeight = new double[receptiveFieldWidth, receptiveFieldHeight];
            InitShareWeight();
        }
        /// <summary>
        /// 计算卷积结果
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
                    //tanh
                    result = ActivationFunctionTanh(value, x, y);
                    break;
                case 2:
                    //平均池化
                    result = ActivationFunctionMeanPooling(value, x, y);
                    break;
                case 3:
                    //最大值池化
                    result = ActivationFunctionMaxPooling(value, x, y);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanh(double[,] value, int x, int y)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    result += value[offsetWidth * x + i, offsetHeight * y + j] * ShareWeight[i, j];
                }
            }
            //调用激活函数计算结果
            result = Math.Tanh(result + OutputOffset);
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
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    result += value[offsetWidth * x + i, offsetHeight * y + j];
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
            double result = 0;
            //计算区域内的最大值
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    if (result < value[offsetWidth * x + 1, offsetHeight * y + j])
                        result = value[offsetWidth * x + 1, offsetHeight * y + j];
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
    }
}

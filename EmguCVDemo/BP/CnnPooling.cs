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
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Ceiling(inputWidth / (double)receptiveFieldWidth));
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Ceiling(inputHeight / (double)receptiveFieldWidth));
            OutputPoolingMax = new int[ConvolutionKernelWidth, ConvolutionKernelHeight];
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
            for (int i = 0; i < receptiveFieldWidth && receptiveFieldWidth * x + i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && receptiveFieldHeight * y + j < value.GetLength(1); j++)
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
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <returns></returns>
        public double[,] BackPropagation(double[,] output)
        {
            double[,] result = null;//正确输入值
            switch (ActivationFunctionType)
            {
                case 1:
                    //平均池化
                    result = CalculatedBackPropagationResultMeanPooling(output);
                    break;
                case 2:
                    //最大值池化
                    result = CalculatedBackPropagationResultMaxPooling(output);
                    break;
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
                    if (OutputPoolingMax[i, j] == receptiveFieldWidth % i + receptiveFieldHeight % j * receptiveFieldHeight)
                        result[i, j] = output[i / receptiveFieldWidth, j / receptiveFieldHeight];
                    else
                        result[i, j] = 0;
                }
            }
            return result;
        }
    }
}

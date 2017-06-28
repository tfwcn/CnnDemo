using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 卷积核(池化)
    /// </summary>
    [Serializable]
    public class CnnPooling : CnnNode
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
        /// 池化类型，1:平均池化(Mean Pooling),2:最大值池化(Max Pooling)
        /// </summary>
        public int PoolingType { get; set; }
        /// <summary>
        /// 原输入值
        /// </summary>
        [NonSerialized]
        public double[,] InputValue;
        /// <summary>
        /// 原输出值
        /// </summary>
        [NonSerialized]
        public double[,] OutputValue;
        /// <summary>
        /// 最大值池化时的最大值指针
        /// </summary>
        private int[,] OutputPoolingMax;
        /// <summary>
        /// 输入数量
        /// </summary>
        public int InputCount { get; set; }
        /// <summary>
        /// 输出数量（该层卷积核数量）
        /// </summary>
        public int OutputCount { get; set; }
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="poolingType">池化类型，1:平均池化(Mean Pooling),2:最大值池化(Max Pooling)</param>
        public CnnPooling(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            int poolingType, int inputCount, int outputCount)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.PoolingType = poolingType;
            this.InputCount = inputCount;
            this.OutputCount = outputCount;
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
                    result[i, j] = CalculatedConvolutionPointResult(value, i, j);//卷积+池化
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
                    result += value[receptiveFieldWidth * x + i, receptiveFieldHeight * y + j];
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
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <returns></returns>
        public double[,] BackPropagation(double[,] output, double learningRate)
        {
            double[,] result = CalculatedBackPropagationResult(output, learningRate);//正确输入值
            return result;
        }
        /// <summary>
        /// 计算反向传播结果
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResult(double[,] output, double learningRate)
        {
            double[,] result = null;//正确输入值
            //调用激活函数计算结果
            switch (PoolingType)
            {
                case 1:
                    //平均池化
                    result = CalculatedBackPropagationResultMeanPooling(output, learningRate);
                    break;
                case 2:
                    //最大值池化
                    result = CalculatedBackPropagationResultMaxPooling(output, learningRate);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（平均池化）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultMeanPooling(double[,] output, double learningRate)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出/感知野大小，放入原来位置
            //计算残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    double residual = OutputValue[i, j] - output[i, j];//输出残差
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldHeight + j2 < inputHeight; j2++)
                        {
                            //计算输入值残差=梯度/卷积核大小，梯度直接往上传
                            result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = InputValue[i, j] - residual / (receptiveFieldWidth * receptiveFieldHeight);
                        }
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（最大值池化）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultMaxPooling(double[,] output, double learningRate)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出放入原来最大值位置，其余位置设置成0
            //计算残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    double residual = OutputValue[i, j] - output[i, j];//输出残差
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldHeight + j2 < inputHeight; j2++)
                        {
                            //计算输入值残差，梯度直接往上传
                            if (OutputPoolingMax[i, j] == i2 + j2 * receptiveFieldHeight)
                            {
                                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = InputValue[i, j] - residual;
                            }
                            else
                                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = 0;
                        }
                    }
                }
            }
            return result;
        }
    }
}

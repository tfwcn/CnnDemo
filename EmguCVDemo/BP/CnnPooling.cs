using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 卷积核(池化)
    /// </summary>
    public class CnnPooling : CnnNode
    {
        /// <summary>
        /// 共享权重(所有感知野共享，所有权重一样)
        /// </summary>
        public double ShareWeight { get; set; }
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
                    result += value[receptiveFieldWidth * x + i, receptiveFieldHeight * y + j] * ShareWeight;
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
            result *= ShareWeight;
            return result;
        }
        /// <summary>
        /// 初始化共享权重和偏置
        /// </summary>
        private void InitShareWeight()
        {
            Random random = new Random();
            ShareWeight = GetRandom(random);
            OutputOffset = GetRandom(random);
        }
        /// <summary>
        /// 获取随机值
        /// </summary>
        private double GetRandom(Random random)
        {
            double result = 0;
            int inputCount = inputWidth;
            int outputCount = ConvolutionKernelWidth;
            switch (ActivationFunctionType)
            {
                case 2:
                    //PReLU
                    result = random.NextDouble() * 0.0001;
                    break;
                default:
                    result = 1;
                    break;
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
            //输出残差
            double[,] residual = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            //权重残差
            double deltaWeight = 0;
            //偏置残差
            double deltaOffset = 0;
            //残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    residual[i, j] = OutputValue[i, j] - output[i, j];
                }
            }
            //计算残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldHeight + j2 < inputHeight; j2++)
                        {
                            //计算输入值残差
                            result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] += ShareWeight * residual[i, j];
                            //计算权重残差
                            deltaWeight += InputValue[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] * residual[i, j];
                        }
                    }
                    //计算偏置残差
                    deltaOffset += residual[i, j];
                }
            }
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] /= receptiveFieldWidth * receptiveFieldHeight;
                    result[i, j] *= ActivationFunctionDerivative(InputValue[i, j]);//旧代码没有
                }
            }
            deltaWeight /= receptiveFieldWidth * receptiveFieldHeight;
            //更新权重和偏置
            UpdateWeight(ShareWeight, deltaWeight, learningRate);
            UpdateOffset(OutputOffset, deltaOffset, learningRate);
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（最大值池化）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultMaxPooling(double[,] output, double learningRate)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出放入原来最大值位置，其余位置设置成0
            //输出残差
            double[,] residual = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            //权重残差
            double deltaWeight = 0;
            //偏置残差
            double deltaOffset = 0;
            //残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    residual[i, j] = OutputValue[i, j] - output[i, j];
                }
            }
            //计算残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldHeight + j2 < inputHeight; j2++)
                        {
                            //计算输入值残差
                            if (OutputPoolingMax[ConvolutionKernelWidth, ConvolutionKernelWidth] == i2 + j2 * receptiveFieldHeight)
                            {
                                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] += ShareWeight * residual[i, j];
                                //计算权重残差
                                deltaWeight += InputValue[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] * residual[i, j];
                            }
                            else
                                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = 0;
                        }
                    }
                    //计算偏置残差
                    deltaOffset += residual[i, j];
                }
            }
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] *= ActivationFunctionDerivative(InputValue[i, j]);//旧代码没有
                }
            }
            //更新权重和偏置
            UpdateWeight(ShareWeight, deltaWeight, learningRate);
            UpdateOffset(OutputOffset, deltaOffset, learningRate);
            return result;
        }
        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="weight">权重</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateWeight(double weight, double delta, double learningRate)
        {
            //weight -= learningRate * delta / (delta + 1e-8);
            weight -= learningRate * delta;//旧
        }
        /// <summary>
        /// 更新偏置
        /// </summary>
        /// <param name="weight">偏置</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateOffset(double offset, double delta, double learningRate)
        {
            //offset -= learningRate * delta / (delta + 1e-8);
            offset -= learningRate * delta;//旧
        }
    }
}

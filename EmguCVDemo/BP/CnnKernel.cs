using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 卷积核
    /// </summary>
    [Serializable]
    public class CnnKernel : CnnNode
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
        /// 偏移值（宽）
        /// </summary>
        private int offsetWidth;
        /// <summary>
        /// 偏移值（高）
        /// </summary>
        private int offsetHeight;
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
        /// <param name="offsetWidth"></param>
        /// <param name="offsetHeight"></param>
        /// <param name="activationFunctionType">激活函数类型，1:tanh,2:PReLU,3:Sigmoid</param>
        /// <param name="inputCount"></param>
        /// <param name="outputCount"></param>
        public CnnKernel(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int activationFunctionType, int inputCount, int outputCount)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.offsetWidth = offsetWidth;
            this.offsetHeight = offsetHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.InputCount = inputCount;
            this.OutputCount = outputCount;
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
                    result[i, j] = CalculatedConvolutionPointResult(value, i, j);//卷积
                    result[i, j] += OutputOffset;//偏置
                    //调用激活函数计算结果
                    result[i, j] = ActivationFunction(result[i, j]);
                }
            }
            //double mean = CnnHelper.GetMean(result);
            //double variance = CnnHelper.GetVariance(result, mean);
            ////归一化每个结果
            //for (int i = 0; i < ConvolutionKernelWidth; i++)
            //{
            //    for (int j = 0; j < ConvolutionKernelHeight; j++)
            //    {
            //        double z = (result[i, j] - mean) / Math.Sqrt(variance);
            //        //调用激活函数计算结果
            //        result[i, j] = ActivationFunction(z * Math.Sqrt(variance) + mean);
            //    }
            //}
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算感知野结果(卷积)
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
            OutputOffset = GetRandom(random);
        }
        /// <summary>
        /// 获取随机值
        /// </summary>
        private double GetRandom(Random random)
        {
            double result = 0;
            switch (ActivationFunctionType)
            {
                case 2:
                    //PReLU
                    result = random.NextDouble() * 0.0001;
                    break;
                default:
                    result = (random.NextDouble() * 2 - 1) * Math.Sqrt((float)6.0 / (float)(receptiveFieldWidth * receptiveFieldHeight * (InputCount + OutputCount)));
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
            result = CalculatedBackPropagationResult(output, learningRate);
            return result;
        }
        /// <summary>
        /// 计算反向传播结果
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResult(double[,] output, double learningRate)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值
            //输出残差
            double[,] residual = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            //权重残差
            double[,] deltaWeight = new double[receptiveFieldWidth, receptiveFieldHeight];
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
            //result = CnnHelper.Convolution(ShareWeight, residual, offsetWidth, offsetHeight, 1);
            //double[,] result2 = CnnHelper.Convolution(ShareWeight, residual, offsetWidth, offsetHeight, 1);
            //double[,] deltaWeight2 = CnnHelper.Convolution(residual, InputValue, offsetWidth, offsetHeight, 3);
            //deltaWeight2 = CnnHelper.MatrixRotate180(deltaWeight2);
            //计算残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * offsetWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * offsetHeight + j2 < inputHeight; j2++)
                        {
                            //计算输入值残差
                            result[i * offsetWidth + i2, j * offsetHeight + j2] += ShareWeight[i2, j2] * residual[i, j];// / (receptiveFieldWidth * receptiveFieldHeight));
                            //计算权重残差
                            deltaWeight[i2, j2] += InputValue[i * offsetWidth + i2, j * offsetHeight + j2] * residual[i, j];
                        }
                    }
                    //计算偏置残差
                    deltaOffset += residual[i, j];
                }
            }
            deltaWeight = CnnHelper.MatrixRotate180(deltaWeight);
            //更新权重和偏置
            UpdateWeight(ShareWeight, deltaWeight, learningRate);
            UpdateOffset(OutputOffset, deltaOffset, learningRate);
            //计算正确输入值
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] *= ActivationFunctionDerivative(InputValue[i, j]);
                    result[i, j] = InputValue[i, j] - result[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="weight">权重</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateWeight(double[,] weight, double[,] delta, double learningRate)
        {
            for (int i = 0; i < receptiveFieldWidth; i++)
            {
                for (int j = 0; j < receptiveFieldHeight; j++)
                {
                    weight[i, j] -= learningRate * delta[i, j];
                }
            }
        }
        /// <summary>
        /// 更新偏置
        /// </summary>
        /// <param name="weight">偏置</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateOffset(double offset, double delta, double learningRate)
        {
            offset -= learningRate * delta;
        }
    }
}

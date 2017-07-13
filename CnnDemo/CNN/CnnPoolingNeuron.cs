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
    public class CnnPoolingNeuron : CnnNeuron
    {
        /// <summary>
        /// 池化类型
        /// </summary>
        public enum PoolingTypes
        {
            /// <summary>
            /// 无池化层
            /// </summary>
            None = 0,
            /// <summary>
            /// 平均池化
            /// </summary>
            MeanPooling = 1,
            /// <summary>
            /// 最大值池化
            /// </summary>
            MaxPooling = 2
        }
        /// <summary>
        /// 共享权重,所有权重同一个值
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
        public PoolingTypes PoolingType { get; set; }
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
        /// 共享权重梯度集，用于计算平均权重梯度
        /// </summary>
        private List<double> meanListDeltaWeight;
        /// <summary>
        /// 偏置梯度集，用于计算平均偏置梯度
        /// </summary>
        private List<double> meanListDeltaOffset;
        /// <summary>
        /// 平均共享权重梯度
        /// </summary>
        private double meanDeltaWeight;
        /// <summary>
        /// 平均偏置梯度
        /// </summary>
        private double meanDeltaOffset;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="poolingType">池化类型，1:平均池化(Mean Pooling),2:最大值池化(Max Pooling)</param>
        public CnnPoolingNeuron(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            ActivationFunctionTypes activationFunctionType, PoolingTypes poolingType, int inputCount, int outputCount)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.PoolingType = poolingType;
            this.InputCount = inputCount;
            this.OutputCount = outputCount;
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Ceiling(inputWidth / (double)receptiveFieldWidth));
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Ceiling(inputHeight / (double)receptiveFieldWidth));
            if (this.ConvolutionKernelWidth <= 0 || this.ConvolutionKernelHeight <= 0)
                throw new Exception("卷积核大小有误");
            OutputPoolingMax = new int[ConvolutionKernelWidth, ConvolutionKernelHeight];
            meanListDeltaWeight = new List<double>();
            meanListDeltaOffset = new List<double>();
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
                    result[i, j] = ActivationFunction(ShareWeight * result[i, j] + OutputOffset);//权重、偏置
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
                case PoolingTypes.MeanPooling:
                    //平均池化
                    result = MeanPooling(value, x, y);
                    break;
                case PoolingTypes.MaxPooling:
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
        /// 初始化共享权重和偏置
        /// </summary>
        private void InitShareWeight()
        {
            //Random random = new Random();
            Random random = CnnHelper.RandomObj;
            //ShareWeight = 1;
            ShareWeight = GetRandom(random);
            //OutputOffset = GetRandom(random);
        }
        /// <summary>
        /// 获取随机值
        /// </summary>
        private double GetRandom(Random random)
        {
            double result = 0;
            switch (ActivationFunctionType)
            {
                case ActivationFunctionTypes.ReLU:
                    //PReLU
                    //result = random.NextDouble() * 0.0001;
                    result = 1 + (random.NextDouble() - 0.5);
                    break;
                default:
                    //result = (random.NextDouble() * 2 - 1) * Math.Sqrt((float)6.0 / (float)(receptiveFieldWidth * receptiveFieldHeight * (InputCount + OutputCount)));
                    result = 1 + (random.NextDouble() - 0.5);
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
        public double[,] BackPropagation(double[,] residual, double learningRate)
        {
            //当前层残差
            double[,] residualNow = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            //权重残差
            double deltaWeight = 0;
            //偏置残差
            double deltaOffset = 0;
            //残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    residualNow[i, j] = ActivationFunctionDerivative(OutputValue[i, j]) * residual[i, j];//CNN标准
                    deltaWeight += residualNow[i, j] * OutputValue[i, j];//CNN标准
                    deltaOffset += residualNow[i, j];//CNN标准
                }
            }
            //计算上一层残差
            double[,] result = CalculatedBackPropagationResult(residualNow);//反池化，向上采样
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] *= ActivationFunctionDerivative(InputValue[i, j]) * ShareWeight;//CNN标准
                }
            }
            result = CnnHelper.MatrixCut(result, 0, 0, result.GetLength(0) - inputWidth, result.GetLength(1) - inputHeight);//裁剪
            //计算平均梯度
            //*
            meanListDeltaWeight.Add(deltaWeight);
            if (meanListDeltaWeight.Count > MiniBatchSize)
            {
                meanDeltaWeight -= meanListDeltaWeight[0] / MiniBatchSize;
                meanDeltaWeight += deltaWeight / MiniBatchSize;
                meanListDeltaWeight.RemoveAt(0);
            }
            else
            {
                meanDeltaWeight = 0;
                foreach (var tmpShareWeight in meanListDeltaWeight)
                {
                    meanDeltaWeight += tmpShareWeight / meanListDeltaWeight.Count;
                }
            }
            meanListDeltaOffset.Add(deltaOffset);
            if (meanListDeltaOffset.Count > MiniBatchSize)
            {
                meanDeltaOffset -= meanListDeltaOffset[0] / MiniBatchSize;
                meanDeltaOffset += deltaOffset / MiniBatchSize;
                meanListDeltaOffset.RemoveAt(0);
            }
            else
            {
                meanDeltaOffset = 0;
                foreach (var tmpShareOffset in meanListDeltaOffset)
                {
                    meanDeltaOffset += tmpShareOffset / meanListDeltaOffset.Count;
                }
            }
            //*/
            //更新权重和偏置
            //UpdateWeight(deltaWeight, learningRate);//CNN标准
            //UpdateOffset(deltaOffset, learningRate);//CNN标准
            UpdateWeight(meanDeltaWeight, learningRate);
            UpdateOffset(meanDeltaOffset, learningRate);
            return result;
        }
        /// <summary>
        /// 计算反向传播结果
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResult(double[,] output)
        {
            double[,] result = null;//正确输入值
            //调用激活函数计算结果
            switch (PoolingType)
            {
                case PoolingTypes.MeanPooling:
                    //平均池化
                    result = CalculatedBackPropagationResultMeanPooling(output);
                    break;
                case PoolingTypes.MaxPooling:
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
        private double[,] CalculatedBackPropagationResultMeanPooling(double[,] residual)
        {
            //double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出/感知野大小，放入原来位置
            ////计算残差
            //for (int i = 0; i < ConvolutionKernelWidth; i++)
            //{
            //    for (int j = 0; j < ConvolutionKernelHeight; j++)
            //    {
            //        for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
            //        {
            //            for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldHeight + j2 < inputHeight; j2++)
            //            {
            //                //计算输入值残差=梯度/卷积核大小，梯度直接往上传
            //                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = residual[i, j] / (receptiveFieldWidth * receptiveFieldHeight);
            //            }
            //        }
            //    }
            //}
            double[,] result = CnnHelper.MatrixScale(residual, receptiveFieldWidth, receptiveFieldHeight);
            for (int i = 0; i < inputWidth; i++)
            {
                for (int j = 0; j < inputHeight; j++)
                {
                    result[i, j] /= receptiveFieldWidth * receptiveFieldHeight;//取平均
                }
            }
            return result;
        }
        /// <summary>
        /// 计算反向传播结果（最大值池化）
        /// </summary>
        /// <returns></returns>
        private double[,] CalculatedBackPropagationResultMaxPooling(double[,] residual)
        {
            double[,] result = new double[inputWidth, inputHeight];//正确输入值=正确输出放入原来最大值位置，其余位置设置成0
            //计算残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    for (int i2 = 0; i2 < receptiveFieldWidth && i * receptiveFieldWidth + i2 < inputWidth; i2++)
                    {
                        for (int j2 = 0; j2 < receptiveFieldHeight && j * receptiveFieldHeight + j2 < inputHeight; j2++)
                        {
                            //计算输入值残差，梯度直接往上传
                            if (OutputPoolingMax[i, j] == i2 + j2 * receptiveFieldHeight)
                            {
                                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = residual[i, j];
                            }
                            else
                                result[i * receptiveFieldWidth + i2, j * receptiveFieldHeight + j2] = 0;
                        }
                    }
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
        private void UpdateWeight(double delta, double learningRate)
        {
            ShareWeight += learningRate * delta;
        }
        /// <summary>
        /// 更新偏置
        /// </summary>
        /// <param name="weight">偏置</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateOffset(double delta, double learningRate)
        {
            OutputOffset += learningRate * delta;
        }
    }
}

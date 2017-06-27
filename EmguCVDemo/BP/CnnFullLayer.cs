using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 全链接层
    /// </summary>
    [Serializable]
    public class CnnFullLayer : CnnNode
    {
        /// <summary>
        /// 权重
        /// </summary>
        public double[,] InputWeight { get; set; }
        /// <summary>
        /// 偏置
        /// </summary>
        public double[] OutputOffset { get; set; }
        /// <summary>
        /// 输入数量
        /// </summary>
        public int InputCount { get; set; }
        /// <summary>
        /// 输出数量(神经元数量)
        /// </summary>
        public int OutputCount { get; set; }
        /// <summary>
        /// 原输入值
        /// </summary>
        [NonSerialized]
        public double[] InputValue;
        /// <summary>
        /// 原输出值
        /// </summary>
        [NonSerialized]
        public double[] OutputValue;
        /// <summary>
        /// 输出平均值
        /// </summary>
        private double mean;
        /// <summary>
        /// 输出方差
        /// </summary>
        private double variance;
        /// <summary>
        /// 归一化
        /// </summary>
        public bool Standardization { get; set; }
        /// <summary>
        /// 共享权重梯度集，用于计算平均权重梯度
        /// </summary>
        private List<double[,]> meanListDeltaWeight;
        /// <summary>
        /// 偏置梯度集，用于计算平均偏置梯度
        /// </summary>
        private List<double[]> meanListDeltaOffset;
        /// <summary>
        /// 平均共享权重梯度
        /// </summary>
        private double[,] meanDeltaWeight;
        /// <summary>
        /// 平均偏置梯度
        /// </summary>
        private double[] meanDeltaOffset;
        /// <summary>
        /// 平均梯度集上限
        /// </summary>
        private int miniBatchSize = 10;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="InputCount"></param>
        /// <param name="OutputCount"></param>
        /// <param name="activationFunctionType">激活函数类型，1:tanh,2:PReLU,3:Sigmoid</param>
        public CnnFullLayer(int InputCount, int OutputCount, int activationFunctionType, bool standardization)
        {
            this.InputCount = InputCount;
            this.OutputCount = OutputCount;
            this.ActivationFunctionType = activationFunctionType;
            this.Standardization = standardization;
            InputWeight = new double[OutputCount, InputCount];
            OutputOffset = new double[OutputCount];
            meanDeltaWeight = new double[OutputCount, InputCount];
            meanDeltaOffset = new double[OutputCount];
            meanListDeltaWeight = new List<double[,]>();
            meanListDeltaOffset = new List<double[]>();
            InitInputWeight();
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public double[] CalculatedResult(double[] value)
        {
            InputValue = value;
            double[] result = new double[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                result[i] = CalculatedPointResult(value, i);
                //result[i] = CalculatedPointResult(value, i) + OutputOffset[i];
                //调用激活函数计算结果
                if (!Standardization)
                    result[i] = ActivationFunction(result[i] + OutputOffset[i]);
            }
            if (Standardization)
            {
                mean = CnnHelper.GetMean(result);
                variance = CnnHelper.GetVariance(result, mean);
                //归一化每个结果
                for (int i = 0; i < OutputCount; i++)
                {
                    //调用激活函数计算结果
                    double z = (result[i] - mean) / Math.Sqrt(variance);
                    result[i] = ActivationFunction(z + OutputOffset[i]);
                }
            }
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算单个神经元结果
        /// </summary>
        /// <returns></returns>
        private double CalculatedPointResult(double[] value, int index)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < InputCount; i++)
            {
                result += value[i] * InputWeight[index, i];
            }
            return result;
        }
        /// <summary>
        /// 初始化权重
        /// </summary>
        private void InitInputWeight()
        {
            Random random = new Random();
            for (int i = 0; i < InputWeight.GetLength(0); i++)
            {
                for (int j = 0; j < InputWeight.GetLength(1); j++)
                {
                    InputWeight[i, j] = GetRandom(random);
                }
            }
            for (int i = 0; i < OutputOffset.Length; i++)
            {
                OutputOffset[i] = GetRandom(random);
            }
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
                    result = (random.NextDouble() * 2 - 1) * Math.Sqrt((float)6.0 / (float)(InputCount + OutputCount));
                    break;
            }
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns>返回更新权重后的输入值</returns>
        public double[] BackPropagation(double[] output, double learningRate)
        {
            double[] result = new double[InputCount];
            //权重残差
            double[,] deltaWeight = new double[OutputCount, InputCount];
            //偏置残差
            double[] deltaOffset = new double[OutputCount];
            //计算上一层的残差
            for (int i = 0; i < OutputCount; i++)
            {
                //残差=导数(输出值)*(输出值-正确值)
                //double residual = ActivationFunctionDerivative(OutputValue[i]) * (OutputValue[i] - output[i]);
                double residual = OutputValue[i] - output[i];
                for (int j = 0; j < InputCount; j++)
                {
                    //sum(残差)=更新前的权重*残差
                    result[j] += InputWeight[i, j] * residual;
                    //计算权重残差,sum(残差)=残差*输入值
                    deltaWeight[i, j] += residual * InputValue[j];
                }
                deltaOffset[i] = residual;
            }
            //计算平均梯度
            /*
            meanListDeltaWeight.Add(deltaWeight);
            for (int i = 0; i < OutputCount; i++)
            {
                for (int j = 0; j < InputCount; j++)
                {
                    if (meanListDeltaWeight.Count > miniBatchSize)
                    {
                        meanDeltaWeight[i, j] -= meanListDeltaWeight[0][i, j] / miniBatchSize;
                        meanDeltaWeight[i, j] += deltaWeight[i, j] / miniBatchSize;
                        meanListDeltaWeight.RemoveAt(0);
                    }
                    else
                    {
                        meanDeltaWeight[i, j] = 0;
                        foreach (var tmpShareWeight in meanListDeltaWeight)
                        {
                            meanDeltaWeight[i, j] += tmpShareWeight[i, j] / meanListDeltaWeight.Count;
                        }
                    }
                }
            }
            meanListDeltaOffset.Add(deltaOffset);
            for (int i = 0; i < OutputCount; i++)
            {
                if (meanListDeltaOffset.Count > miniBatchSize)
                {
                    meanDeltaOffset[i] -= meanListDeltaOffset[i][0] / miniBatchSize;
                    meanDeltaOffset[i] += deltaOffset[i] / miniBatchSize;
                    meanListDeltaOffset.RemoveAt(0);
                }
                else
                {
                    meanDeltaOffset[i] = 0;
                    foreach (var tmpShareOffset in meanListDeltaOffset)
                    {
                        meanDeltaOffset[i] += tmpShareOffset[i] / meanListDeltaOffset.Count;
                    }
                }
            }
            //*/
            //更新权重和偏置
            UpdateWeight(InputWeight, deltaWeight, learningRate);
            UpdateOffset(OutputOffset, deltaOffset, learningRate);
            //UpdateWeight(InputWeight, meanDeltaWeight, learningRate);
            //UpdateOffset(OutputOffset, meanDeltaOffset, learningRate);
            //计算正确输入值
            for (int i = 0; i < InputCount; i++)
            {
                //正确输入值=旧输入值-sum(残差*更新前的权重)
                result[i] *= ActivationFunctionDerivative(InputValue[i]);
                result[i] = InputValue[i] - result[i];
                //反归一化每个结果
                if (Standardization)
                    result[i] = result[i] * Math.Sqrt(variance) + mean;
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
            //Console.WriteLine(String.Format("FullUpdateWeight {0}->{1}", InputCount, OutputCount));
            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    weight[j, i] -= learningRate * delta[j, i];
                    //Console.Write(weight[j, i] + " ");
                }
                //Console.WriteLine("");
            }
            //Console.WriteLine("");
        }
        /// <summary>
        /// 更新偏置
        /// </summary>
        /// <param name="weight">偏置</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateOffset(double[] offset, double[] delta, double learningRate)
        {
            //Console.WriteLine(String.Format("FullUpdateOffset {0}->{1}", InputCount, OutputCount));
            for (int i = 0; i < OutputCount; i++)
            {
                offset[i] -= learningRate * delta[i];
                //Console.Write(offset[i] + " ");
            }
            //Console.WriteLine("");
        }
    }
}

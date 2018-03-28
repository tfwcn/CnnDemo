using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 全链接神经元
    /// </summary>
    [Serializable]
    public class CnnFullNeuron : CnnNeuron
    {
        /// <summary>
        /// 权重
        /// </summary>
        public double[] InputWeight { get; set; }
        /// <summary>
        /// 偏置
        /// </summary>
        public double OutputOffset { get; set; }
        /// <summary>
        /// 输入数量
        /// </summary>
        public int InputCount { get; set; }
        /// <summary>
        /// 输出数量(神经元数量)
        /// </summary>
        public int OutputCount { get; private set; }
        /// <summary>
        /// 原输入值
        /// </summary>
        [NonSerialized]
        public double[] InputValue;
        /// <summary>
        /// 原输出值
        /// </summary>
        [NonSerialized]
        public double OutputValue;
        /// <summary>
        /// 共享权重梯度集，用于计算平均权重梯度
        /// </summary>
        private List<double[]> meanListDeltaWeight;
        /// <summary>
        /// 偏置梯度集，用于计算平均偏置梯度
        /// </summary>
        private List<double> meanListDeltaOffset;
        /// <summary>
        /// 平均共享权重梯度
        /// </summary>
        private double[] meanDeltaWeight;
        /// <summary>
        /// 平均偏置梯度
        /// </summary>
        private double meanDeltaOffset;
        /// <summary>
        /// 正则化概率（Dropout）
        /// </summary>
        private double dropoutChance = -1;
        /// <summary>
        /// 正则化状态（Dropout）
        /// </summary>
        private bool dropoutState = false;
        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="InputCount"></param>
        /// <param name="OutputCount"></param>
        /// <param name="activationFunctionType">激活函数类型，1:tanh,2:PReLU,3:Sigmoid</param>
        public CnnFullNeuron(int InputCount, int OutputCount, ActivationFunctionTypes activationFunctionType)
        {
            this.InputCount = InputCount;
            this.OutputCount = OutputCount;
            this.ActivationFunctionType = activationFunctionType;
            InputWeight = new double[InputCount];
            OutputOffset = 0;
            meanDeltaWeight = new double[InputCount];
            meanDeltaOffset = 0;
            meanListDeltaWeight = new List<double[]>();
            meanListDeltaOffset = new List<double>();
            InitInputWeight();
        }
        /// <summary>
        /// 初始化权重
        /// </summary>
        private void InitInputWeight()
        {
            //Random random = new Random();
            Random random = CnnHelper.RandomObj;
            for (int i = 0; i < InputCount; i++)
            {
                InputWeight[i] = GetRandom(random);
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
                case CnnDemo.CNN.CnnNeuron.ActivationFunctionTypes.ReLU:
                    //PReLU
                    //result = random.NextDouble() * 0.0001;
                    result = (random.NextDouble() - 0.5) * Math.Sqrt(6.0 / (InputCount + OutputCount));
                    break;
                default:
                    result = (random.NextDouble() - 0.5) * Math.Sqrt(6.0 / (InputCount + OutputCount));
                    break;
            }
            return result;
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public double CalculatedResult(double[] inputValue)
        {
            InputValue = inputValue;
            double result = 0;
            //正则化
            if (CnnHelper.RandomObj.NextDouble() < dropoutChance)
            {
                result = 0;
                dropoutState = true;
                OutputValue = result;
                return result;
            }
            result = CalculatedPointResult(inputValue);
            //调用激活函数计算结果
            result = ActivationFunction(result + OutputOffset);
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算单个神经元结果
        /// </summary>
        /// <returns></returns>
        private double CalculatedPointResult(double[] value)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < InputCount; i++)
            {
                result += value[i] * InputWeight[i];
            }
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns>返回更新权重后的输入值</returns>
        public double[] BackPropagation(double residual, double learningRate)
        {
            double[] result = new double[InputCount];
            //下一层残差
            double[] resultDelta = new double[InputCount];
            //权重残差
            double[] deltaWeight = new double[InputCount];
            //偏置残差
            double deltaOffset = 0;
            //正则化
            if (dropoutState)
            {
                dropoutState = false;
                return result;
            }
            //计算上一层的残差
            //当前层残差=导数(激活函数前的输出值)*(输出值-正确值)
            double outputValueDerivative = ActivationFunctionDerivative(OutputValue) * residual;
            for (int i = 0; i < InputCount; i++)
            {
                //sum(残差)=更新前的权重*残差
                resultDelta[i] += InputWeight[i] * outputValueDerivative;
                if (Double.IsNaN(resultDelta[i]) || Double.IsInfinity(resultDelta[i]))
                    throw new Exception("NaN");
                //计算权重残差,sum(残差)=残差*输入值
                deltaWeight[i] += outputValueDerivative * InputValue[i];
                if (Double.IsNaN(deltaWeight[i]) || Double.IsInfinity(deltaWeight[i]))
                    throw new Exception("NaN");
            }
            deltaOffset = outputValueDerivative;
            if (Double.IsNaN(deltaOffset) || Double.IsInfinity(deltaOffset))
                throw new Exception("NaN");
            //计算平均梯度
            //*
            meanListDeltaWeight.Add(deltaWeight);
            for (int i = 0; i < InputCount; i++)
            {
                if (meanListDeltaWeight.Count > MiniBatchSize)
                {
                    meanDeltaWeight[i] -= meanListDeltaWeight[0][i] / MiniBatchSize;
                    meanDeltaWeight[i] += deltaWeight[i] / MiniBatchSize;
                    meanListDeltaWeight.RemoveAt(0);
                }
                else
                {
                    meanDeltaWeight[i] = 0;
                    foreach (var tmpShareWeight in meanListDeltaWeight)
                    {
                        meanDeltaWeight[i] += tmpShareWeight[i] / meanListDeltaWeight.Count;
                    }
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
            //UpdateWeight(deltaWeight, learningRate);
            //UpdateOffset(deltaOffset, learningRate);
            UpdateWeight(meanDeltaWeight, learningRate);
            UpdateOffset(meanDeltaOffset, learningRate);
            //计算正确输入值
            for (int i = 0; i < InputCount; i++)
            {
                result[i] = resultDelta[i];//正确
                if (Double.IsNaN(result[i]) || Double.IsInfinity(result[i]))
                    throw new Exception("NaN");
            }
            return result;
        }
        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="weight">权重</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateWeight(double[] delta, double learningRate)
        {
            for (int i = 0; i < InputCount; i++)
            {
                InputWeight[i] += learningRate * delta[i];
                if (Double.IsNaN(InputWeight[i]) || Double.IsInfinity(InputWeight[i]))
                    throw new Exception("NaN");
            }
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
            if (Double.IsNaN(OutputOffset) || Double.IsInfinity(OutputOffset))
                throw new Exception("NaN");
        }
        /// <summary>
        /// 神经元描述
        /// </summary>
        //public override string ToString()
        //{
        //    return String.Format("输入:{0} 输出:{1} 输入残差:{2} 权重残差:{3} 偏置残差:{4}",
        //        InputCount, OutputCount,
        //        CnnHelper.GetMeanAbs(debugResult),
        //        CnnHelper.GetMeanAbs(debugDeltaWeight), CnnHelper.GetMeanAbs(debugDeltaOffset));
        //}
    }
}

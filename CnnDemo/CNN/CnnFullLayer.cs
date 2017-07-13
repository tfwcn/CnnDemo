using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 全链接层
    /// </summary>
    [Serializable]
    public class CnnFullLayer : CnnLayer
    {
        /// <summary>
        /// 全链接神经元
        /// </summary>
        private List<CnnFullNeuron> CnnFullList { get; set; }
        /// <summary>
        /// 输入数量
        /// </summary>
        public int InputCount { get; private set; }
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
        /// 构造函数
        /// </summary>
        /// <param name="InputCount"></param>
        /// <param name="OutputCount"></param>
        /// <param name="activationFunctionType">激活函数类型，1:tanh,2:PReLU,3:Sigmoid</param>
        public CnnFullLayer(int InputCount, int NeuronCount, CnnDemo.CNN.CnnNeuron.ActivationFunctionTypes activationFunctionType, bool standardization)
            : base(CnnLayerTypeEnum.Full, NeuronCount)
        {
            this.InputCount = InputCount;
            this.Standardization = standardization;
            this.CnnFullList = new List<CnnFullNeuron>();
            for (int i = 0; i < NeuronCount; i++)
            {
                CnnFullList.Add(new CnnFullNeuron(InputCount, NeuronCount, activationFunctionType));
            }
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public double[] CalculatedResult(double[] inputValue)
        {
            double[] result = new double[NeuronCount];
            int i = 0;
            foreach (var cnnFull in CnnFullList)
            {
                result[i] = cnnFull.CalculatedResult(inputValue);
                i++;
            }
            if (Standardization)
            {
                mean = CnnHelper.GetMean(result);
                variance = CnnHelper.GetVariance(result, mean);
                //归一化每个结果
                for (i = 0; i < NeuronCount; i++)
                {
                    //调用激活函数计算结果
                    result[i] = (result[i] - mean) / Math.Sqrt(variance);
                }
            }
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns>返回更新权重后的输入值</returns>
        public double[] BackPropagation(double[] residual, double learningRate)
        {
            double[] result = new double[InputCount];
            //计算上一层的残差
            for (int i = 0; i < NeuronCount; i++)
            {
                double[] cnnFullResidual = CnnFullList[i].BackPropagation(residual[i], learningRate);
                for (int j = 0; j < InputCount; j++)
                {
                    result[j] += cnnFullResidual[j];
                }
            }
            //计算正确输入值
            for (int i = 0; i < InputCount; i++)
            {
                //反归一化每个结果
                if (Standardization)
                    result[i] = result[i] * Math.Sqrt(variance) + mean;
                if (Double.IsNaN(result[i]) || Double.IsInfinity(result[i]))
                    throw new Exception("NaN");
            }
            return result;
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

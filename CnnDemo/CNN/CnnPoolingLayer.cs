using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 池化层
    /// </summary>
    [Serializable]
    public class CnnPoolingLayer : CnnLayer
    {
        public object lockObj = new object();
        /// <summary>
        /// 池化神经元
        /// </summary>
        private List<CnnPoolingNeuron> CnnPoolingList { get; set; }
        /// <summary>
        /// 输出宽度
        /// </summary>
        public int OutputWidth { get; private set; }
        /// <summary>
        /// 输出高度
        /// </summary>
        public int OutputHeight { get; private set; }
        /// <summary>
        /// 创建池化层
        /// </summary>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="activationFunctionType"></param>
        public CnnPoolingLayer(int NeuronCount, int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight, CnnNeuron.ActivationFunctionTypes activationFunctionType, CnnPoolingNeuron.PoolingTypes poolingType)
            : base(CnnLayerTypeEnum.Pooling, NeuronCount)
        {
            CnnPoolingList = new List<CnnPoolingNeuron>();
            for (int i = 0; i < NeuronCount; i++)
            {
                CnnPoolingList.Add(new CnnPoolingNeuron(inputWidth, inputHeight,
                    receptiveFieldWidth, receptiveFieldHeight, activationFunctionType, poolingType, 1, NeuronCount));
            }
            this.OutputWidth = CnnPoolingList[0].ConvolutionKernelWidth;
            this.OutputHeight = CnnPoolingList[0].ConvolutionKernelHeight;
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public List<double[,]> CalculatedResult(List<double[,]> value)
        {
            List<double[,]> result = new List<double[,]>();
            //for (int i = 0; i < ConvolutionKernelCount; i++)
            //{
            //    result.Add(CnnPoolingList[i].CalculatedConvolutionResult(value[i]));
            //}

            //保证顺序
            for (int i = 0; i < NeuronCount; i++)
            {
                result.Add(null);
            }
            //多线程
            System.Threading.Tasks.Parallel.For(0, NeuronCount, i =>
            {
                try
                {
                    var tmpResult = CnnPoolingList[i].CalculatedConvolutionResult(value[i]);
                    System.Threading.Monitor.Enter(lockObj);
                    result[i] = tmpResult;
                    System.Threading.Monitor.Exit(lockObj);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                    throw ex;
                }
            });
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns>返回更新权重后的输入值,每个神经元对应的输入神经元的残差</returns>
        public List<double[,]> BackPropagation(List<double[,]> output, double learningRate)
        {
            List<double[,]> result = new List<double[,]>();
            //for (int i = 0; i < ConvolutionKernelCount; i++)
            //{
            //    result.Add(CnnPoolingList[i].BackPropagation(output[i], learningRate));
            //}

            //保证顺序
            for (int i = 0; i < NeuronCount; i++)
            {
                result.Add(null);
            }
            //多线程
            System.Threading.Tasks.Parallel.For(0, NeuronCount, i =>
            {
                try
                {
                    var tmpResult = CnnPoolingList[i].BackPropagation(output[i], learningRate);
                    System.Threading.Monitor.Enter(lockObj);
                    result[i] = tmpResult;
                    System.Threading.Monitor.Exit(lockObj);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                    throw ex;
                }
            });
            return result;
        }
    }
}

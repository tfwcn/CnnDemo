using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 卷积层
    /// </summary>
    [Serializable]
    public class CnnConvolutionLayer : CnnLayer
    {
        public object lockObj = new object();
        /// <summary>
        /// 卷积神经元
        /// </summary>
        private List<CnnConvolutionNeuron> CnnKernelList { get; set; }
        /// <summary>
        /// 输出宽度
        /// </summary>
        public int OutputWidth { get; private set; }
        /// <summary>
        /// 输出高度
        /// </summary>
        public int OutputHeight { get; private set; }
        /// <summary>
        /// 与上一层的链接方式
        /// </summary>
        public bool[,] LayerLinks { get; private set; }
        /// <summary>
        /// 创建卷积层
        /// </summary>
        /// <param name="convolutionKernelCount"></param>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="offsetWidth"></param>
        /// <param name="offsetHeight"></param>
        /// <param name="activationFunctionType"></param>
        public CnnConvolutionLayer(int NeuronCount, int inputWidth, int inputHeight,
            int receptiveFieldWidth, int receptiveFieldHeight, int offsetWidth, int offsetHeight,
             int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            CnnNeuron.ActivationFunctionTypes activationFunctionType, int LastLayerCount, bool standardization, bool[,] layerLinks)
            : base(CnnLayerTypeEnum.Convolution, NeuronCount)
        {
            CnnKernelList = new List<CnnConvolutionNeuron>();
            for (int i = 0; i < NeuronCount; i++)
            {
                int inputCount = 0;//单个卷积神经元输入数量
                for (int j = 0; j < LastLayerCount; j++)
                {
                    if (layerLinks == null)
                    {
                        inputCount = 1;
                        break;
                    }
                    if (layerLinks[i, j])
                        inputCount++;
                }
                CnnKernelList.Add(new CnnConvolutionNeuron(inputWidth, inputHeight,
                    receptiveFieldWidth, receptiveFieldHeight,
                    offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    activationFunctionType, inputCount, NeuronCount, standardization));
            }
            this.OutputWidth = CnnKernelList[0].ConvolutionKernelWidth;
            this.OutputHeight = CnnKernelList[0].ConvolutionKernelHeight;
            this.LayerLinks = layerLinks;
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public List<double[,]> CalculatedResult(List<List<double[,]>> value)
        {
            List<double[,]> result = new List<double[,]>();
            //for (int i = 0; i < ConvolutionKernelCount; i++)
            //{
            //    if (CnnPoolingList != null)
            //    {
            //        result.Add(CnnPoolingList[i].CalculatedConvolutionResult(CnnKernelList[i].CalculatedConvolutionResult(value[i])));
            //    }
            //    else
            //    {
            //        result.Add(CnnKernelList[i].CalculatedConvolutionResult(value[i]));
            //    }
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
                    var tmpResult = CnnKernelList[i].CalculatedConvolutionResult(value[i]);
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
        public List<List<double[,]>> BackPropagation(List<double[,]> output, double learningRate)
        {
            List<List<double[,]>> result = new List<List<double[,]>>();
            //for (int i = 0; i < ConvolutionKernelCount; i++)
            //{
            //    if (CnnPoolingList != null)
            //    {
            //        result.Add(CnnKernelList[i].BackPropagation(CnnPoolingList[i].BackPropagation(output[i], learningRate), learningRate));
            //    }
            //    else
            //    {
            //        result.Add(CnnKernelList[i].BackPropagation(output[i], learningRate));
            //    }
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
                    var tmpResult = CnnKernelList[i].BackPropagation(output[i], learningRate);
                    System.Threading.Monitor.Enter(lockObj);
                    result[i] = tmpResult;
                    System.Threading.Monitor.Exit(lockObj);
                    //LogHelper.Info(CnnKernelList[i].ToString());
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace CnnDemo.CNN
{
    [Serializable]
    public class CnnConvolutionLayer
    {
        public object lockObj = new object();
        /// <summary>
        /// 卷积层
        /// </summary>
        private List<CnnKernel> CnnKernelList { get; set; }
        /// <summary>
        /// 池化层
        /// </summary>
        private List<CnnPooling> CnnPoolingList { get; set; }
        /// <summary>
        /// 卷积核数量
        /// </summary>
        public int ConvolutionKernelCount { get; set; }
        /// <summary>
        /// 输出宽度
        /// </summary>
        public int OutputWidth
        {
            get
            {
                if (CnnPoolingList != null)
                {
                    return CnnPoolingList[0].ConvolutionKernelWidth;
                }
                else
                {
                    return CnnKernelList[0].ConvolutionKernelWidth;
                }
            }
        }
        /// <summary>
        /// 输出高度
        /// </summary>
        public int OutputHeight
        {
            get
            {
                if (CnnPoolingList != null)
                {
                    return CnnPoolingList[0].ConvolutionKernelHeight;
                }
                else
                {
                    return CnnKernelList[0].ConvolutionKernelWidth;
                }
            }
        }
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
        public void CreateCnnKernel(int convolutionKernelCount, int inputWidth, int inputHeight,
            int receptiveFieldWidth, int receptiveFieldHeight, int offsetWidth, int offsetHeight,
             int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            CnnNode.ActivationFunctionTypes activationFunctionType, int LastLayerCount, bool standardization, bool[,] layerLinks)
        {
            this.ConvolutionKernelCount = convolutionKernelCount;
            CnnKernelList = new List<CnnKernel>();
            for (int i = 0; i < ConvolutionKernelCount; i++)
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
                CnnKernelList.Add(new CnnKernel(inputWidth, inputHeight,
                    receptiveFieldWidth, receptiveFieldHeight,
                    offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    activationFunctionType, inputCount, ConvolutionKernelCount, standardization));
            }
        }
        /// <summary>
        /// 创建池化层
        /// </summary>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="activationFunctionType"></param>
        public void CreateCnnPooling(int receptiveFieldWidth, int receptiveFieldHeight, CnnNode.ActivationFunctionTypes activationFunctionType, CnnPooling.PoolingTypes poolingType)
        {
            if (CnnKernelList == null || CnnKernelList.Count == 0)
                throw new Exception("需先创建卷积层");
            CnnPoolingList = new List<CnnPooling>();
            for (int i = 0; i < ConvolutionKernelCount; i++)
            {
                CnnPoolingList.Add(new CnnPooling(CnnKernelList[0].ConvolutionKernelWidth, CnnKernelList[0].ConvolutionKernelHeight,
                    receptiveFieldWidth, receptiveFieldHeight, activationFunctionType, poolingType, CnnKernelList[0].InputCount, CnnKernelList[0].OutputCount));
            }
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public List<double[,]> CalculatedResult(List<List<double[,]>> value)
        {
            List<double[,]> result = new List<double[,]>();
            //保证顺序
            for (int i = 0; i < ConvolutionKernelCount; i++)
            {
                result.Add(null);
            }
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

            //多线程
            System.Threading.Tasks.Parallel.For(0, ConvolutionKernelCount, i =>
            {
                try
                {
                    if (CnnPoolingList != null)
                    {
                        var tmpResult = CnnPoolingList[i].CalculatedConvolutionResult(CnnKernelList[i].CalculatedConvolutionResult(value[i]));
                        System.Threading.Monitor.Enter(lockObj);
                        result[i] = tmpResult;
                        System.Threading.Monitor.Exit(lockObj);
                    }
                    else
                    {
                        var tmpResult = CnnKernelList[i].CalculatedConvolutionResult(value[i]);
                        System.Threading.Monitor.Enter(lockObj);
                        result[i] = tmpResult;
                        System.Threading.Monitor.Exit(lockObj);
                    }
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
            //保证顺序
            for (int i = 0; i < ConvolutionKernelCount; i++)
            {
                result.Add(null);
            }
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

            //多线程
            System.Threading.Tasks.Parallel.For(0, ConvolutionKernelCount, i =>
            {
                try
                {
                    if (CnnPoolingList != null)
                    {
                        var tmpResult = CnnKernelList[i].BackPropagation(CnnPoolingList[i].BackPropagation(output[i], learningRate), learningRate);
                        System.Threading.Monitor.Enter(lockObj);
                        result[i] = tmpResult;
                        System.Threading.Monitor.Exit(lockObj);
                        LogHelper.Info(CnnKernelList[i].ToString());
                    }
                    else
                    {
                        var tmpResult = CnnKernelList[i].BackPropagation(output[i], learningRate);
                        System.Threading.Monitor.Enter(lockObj);
                        result[i] = tmpResult;
                        System.Threading.Monitor.Exit(lockObj);
                        LogHelper.Info(CnnKernelList[i].ToString());
                    }
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 神经网络组
    /// </summary>
    public class CnnGroup
    {
        public object lockObj = new object();
        /// <summary>
        /// 神经网络集合
        /// </summary>
        private List<Cnn> cnnList = new List<Cnn>();
        /// <summary>
        /// 输出结果数
        /// </summary>
        private int outputCount;

        public CnnGroup(int outputCount)
        {
            this.outputCount = outputCount;
            for (int i = 0; i < outputCount; i++)
            {
                Cnn cnn = new Cnn();
                cnnList.Add(cnn);
            }
        }
        /// <summary>
        /// 增加首个卷积层
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            CnnNode.ActivationFunctionTypes activationFunctionType,
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, CnnPooling.PoolingTypes poolingType,
            bool standardization)
        {
            foreach (Cnn cnn in cnnList)
            {
                cnn.AddCnnConvolutionLayer(convolutionKernelCount,
                    inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight,
                    offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    activationFunctionType,
                    poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingType,
                    standardization);
            }
        }
        /// <summary>
        /// 增加后续卷积层（最后一层一定为单个输出，代表输出中的其中一个值）
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            CnnNode.ActivationFunctionTypes activationFunctionType,
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, CnnPooling.PoolingTypes poolingType,
            bool standardization, bool isFullLayerLinks)
        {
            foreach (Cnn cnn in cnnList)
            {
                cnn.AddCnnConvolutionLayer(convolutionKernelCount,
                    receptiveFieldWidth, receptiveFieldHeight,
                    offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    activationFunctionType,
                    poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingType,
                    standardization, isFullLayerLinks);
            }
        }
        /// <summary>
        /// 增加全连接层，在卷积层后，要先创建完卷积层
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int outputCount, CnnNode.ActivationFunctionTypes activationFunctionType, bool standardization)
        {
            foreach (Cnn cnn in cnnList)
            {
                cnn.AddCnnFullLayer(outputCount, activationFunctionType, standardization);
            }
        }
        /// <summary>
        /// 增加全连接层,仅用于BP网络
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int inputCount, int outputCount, CnnNode.ActivationFunctionTypes activationFunctionType, bool standardization)
        {
            foreach (Cnn cnn in cnnList)
            {
                cnn.AddCnnFullLayer(inputCount, outputCount, activationFunctionType, standardization);
            }
        }
        /// <summary>
        /// 训练
        /// </summary>
        public List<List<List<double[,]>>> Train(double[,] input, double[] output, double learningRate, ref double[] forwardOutputFull)
        {
            List<List<List<double[,]>>> backwardInputConvolutionList = new List<List<List<double[,]>>>();
            List<double[]> forwardOutputFullList = new List<double[]>();
            //保证顺序
            foreach (Cnn cnn in cnnList)
            {
                backwardInputConvolutionList.Add(null);
                forwardOutputFullList.Add(null);
            }
            //多线程
            System.Threading.Tasks.Parallel.For(0, cnnList.Count, i =>
            {
                try
                {
                    double[] forwardOutputFullOne = null;
                    var tmpResult = cnnList[i].Train(input, new double[] { output[i] }, learningRate, ref forwardOutputFullOne);
                    System.Threading.Monitor.Enter(lockObj);
                    backwardInputConvolutionList[i] = tmpResult;
                    forwardOutputFullList[i] = forwardOutputFullOne;
                    System.Threading.Monitor.Exit(lockObj);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                    throw ex;
                }
            });
            //计算全连接层输出
            forwardOutputFull = new double[outputCount];
            for (int i = 0; i < outputCount; i++)
            {
                forwardOutputFull[i] = forwardOutputFullList[i][0];
            }
            return backwardInputConvolutionList;
        }
        /// <summary>
        /// 训练,仅用于BP网络
        /// </summary>
        public List<double[]> TrainFullLayer(double[] input, double[] output, double learningRate)
        {
            List<double[]> backwardInputFullList = new List<double[]>();
            //保证顺序
            foreach (Cnn cnn in cnnList)
            {
                backwardInputFullList.Add(null);
            }
            //多线程
            System.Threading.Tasks.Parallel.For(0, cnnList.Count, i =>
            {
                try
                {
                    var tmpResult = cnnList[i].TrainFullLayer(input, new double[] { output[i] }, learningRate);//每个神经网络对应单个结果
                    System.Threading.Monitor.Enter(lockObj);
                    backwardInputFullList[i] = tmpResult;
                    System.Threading.Monitor.Exit(lockObj);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                    throw ex;
                }
            });
            return backwardInputFullList;
        }
        /// <summary>
        /// 识别
        /// </summary>
        public double[] Predict(double[,] input)
        {
            List<double[]> forwardInputFullList = new List<double[]>();
            //保证顺序
            foreach (Cnn cnn in cnnList)
            {
                forwardInputFullList.Add(null);
            }
            //多线程
            System.Threading.Tasks.Parallel.For(0, cnnList.Count, i =>
            {
                try
                {
                    var tmpResult = cnnList[i].Predict(input);
                    System.Threading.Monitor.Enter(lockObj);
                    forwardInputFullList[i] = tmpResult;
                    System.Threading.Monitor.Exit(lockObj);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                    throw ex;
                }
            });
            //计算全连接层输出
            double[] forwardOutputFull = new double[outputCount];
            for (int i = 0; i < outputCount; i++)
            {
                forwardOutputFull[i] = forwardInputFullList[i][0];
            }
            return forwardOutputFull;
        }
        /// <summary>
        /// 识别,仅用于BP网络
        /// </summary>
        public double[] PredictFullLayer(double[] input)
        {
            List<double[]> forwardInputFullList = new List<double[]>();
            //保证顺序
            foreach (Cnn cnn in cnnList)
            {
                forwardInputFullList.Add(null);
            }
            //多线程
            System.Threading.Tasks.Parallel.For(0, cnnList.Count, i =>
            {
                try
                {
                    var tmpResult = cnnList[i].PredictFullLayer(input);
                    System.Threading.Monitor.Enter(lockObj);
                    forwardInputFullList[i] = tmpResult;
                    System.Threading.Monitor.Exit(lockObj);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                    throw ex;
                }
            });
            //计算全连接层输出
            double[] forwardOutputFull = new double[outputCount];
            for (int i = 0; i < outputCount; i++)
            {
                forwardOutputFull[i] = forwardInputFullList[i][0];
            }
            return forwardOutputFull;
        }
    }
}

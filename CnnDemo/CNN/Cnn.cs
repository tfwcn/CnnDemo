using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN
{
    [Serializable]
    public class Cnn
    {
        /// <summary>
        /// 卷积层+池化层
        /// </summary>
        public List<CnnConvolutionLayer> CnnConvolutionLayerList { get; set; }
        /// <summary>
        /// 全链接层
        /// </summary>
        public List<CnnFullLayer> CnnFullLayerList { get; set; }
        /// <summary>
        /// 卷积层间连接，参数：当前层，上一层
        /// </summary>
        private List<bool[,]> convolutionLinkList;
        public Cnn()
        {
            CnnConvolutionLayerList = new List<CnnConvolutionLayer>();
            CnnFullLayerList = new List<CnnFullLayer>();
            convolutionLinkList = new List<bool[,]>();
        }
        /// <summary>
        /// 增加首个卷积层
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight,
            int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            CnnNeuron.ActivationFunctionTypes activationFunctionType,
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, CnnPoolingNeuron.PoolingTypes poolingType,
            bool standardization)
        {
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight,
                offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight, activationFunctionType, 1, standardization, null);

            if (poolingType != 0)
            {
                //创建池化层
                cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, activationFunctionType, poolingType);
            }
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
        }
        /// <summary>
        /// 增加后续卷积层
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight,
            int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            CnnNeuron.ActivationFunctionTypes activationFunctionType,
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, CnnPoolingNeuron.PoolingTypes poolingType,
            bool standardization, bool isFullLayerLinks)
        {
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            bool[,] layerLinks = new bool[convolutionKernelCount, cnnConvolutionLayerLast.ConvolutionKernelCount];
            if (!isFullLayerLinks)
            {
                //随机创建卷积层间连接
                Random random = new Random();
                for (int i = 0; i < convolutionKernelCount; i++)
                {
                    int linkCount = 0;//每层链接数
                    while (linkCount < 2)//确保最低链接数
                    {
                        for (int j = 0; j < cnnConvolutionLayerLast.ConvolutionKernelCount; j++)
                        {
                            if (random.NextDouble() < 0.5)
                                layerLinks[i, j] = false;
                            else
                            {
                                layerLinks[i, j] = true;
                                linkCount++;
                            }
                        }
                    }
                    //排除相同链接
                    for (int j = 0; j < i; j++)
                    {
                        linkCount = 0;//相同链接数
                        for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                        {
                            if (layerLinks[i, k] == layerLinks[j, k])
                                linkCount++;
                        }
                        if (linkCount == cnnConvolutionLayerLast.ConvolutionKernelCount)
                        {
                            i--;
                            break;
                        }
                    }
                }
            }
            else
            {
                //全链接
                for (int i = 0; i < convolutionKernelCount; i++)
                {
                    for (int j = 0; j < cnnConvolutionLayerLast.ConvolutionKernelCount; j++)
                    {
                        layerLinks[i, j] = true;
                    }
                }
            }
            convolutionLinkList.Add(layerLinks);
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, cnnConvolutionLayerLast.OutputWidth, cnnConvolutionLayerLast.OutputHeight,
                receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                activationFunctionType, cnnConvolutionLayerLast.ConvolutionKernelCount, standardization, layerLinks);

            if (poolingType != 0)
            {
                //创建池化层
                cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, activationFunctionType, poolingType);
            }
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
        }
        /// <summary>
        /// 增加全连接层，在卷积层后，要先创建完卷积层
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int outputCount, CnnNeuron.ActivationFunctionTypes activationFunctionType, bool standardization)
        {
            if (CnnFullLayerList.Count == 0)
            {
                //连接卷积层
                var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
                CnnFullLayer cnnFullLayer = new CnnFullLayer(cnnConvolutionLayerLast.ConvolutionKernelCount
                    * cnnConvolutionLayerLast.OutputWidth
                    * cnnConvolutionLayerLast.OutputHeight,
                    outputCount, activationFunctionType, standardization);
                CnnFullLayerList.Add(cnnFullLayer);
            }
            else
            {
                var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的卷积层
                CnnFullLayer cnnFullLayer = new CnnFullLayer(cnnFullLayerLast.OutputCount,
                    outputCount, activationFunctionType, standardization);
                CnnFullLayerList.Add(cnnFullLayer);
            }
        }
        /// <summary>
        /// 增加全连接层,仅用于BP网络
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int inputCount, int outputCount, CnnNeuron.ActivationFunctionTypes activationFunctionType, bool standardization)
        {
            CnnFullLayer cnnFullLayer = new CnnFullLayer(inputCount,
                outputCount, activationFunctionType, standardization);
            CnnFullLayerList.Add(cnnFullLayer);
        }
        /// <summary>
        /// 训练
        /// </summary>
        public List<List<double[,]>> Train(double[,] input, double[] output, double learningRate, ref double[] forwardOutputFull)
        {
            #region 正向传播
            /*
            //计算卷积层输出
            List<double[,]> forwardOutputConvolution = new List<double[,]>();//输出值
            for (int i = 0; i < CnnConvolutionLayerList.Count; i++)
            {
                //List<List<double[,]>> inputTmp = new List<List<double[,]>>();
                List<List<double[,]>> forwardInputConvolution = new List<List<double[,]>>();//输入值
                if (i == 0)//第一层直接输入
                {
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)
                    {
                        forwardInputConvolution.Add(new List<double[,]> { input });
                    }
                    //inputTmp = forwardInputConvolution;
                }
                else//随机链接
                {
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)
                    {
                        List<double[,]> forwardInputConvolutionOne = new List<double[,]>();//一个神经元的输入值
                        for (int k = 0; k < forwardOutputConvolution.Count; k++)
                        {
                            if (convolutionLinkList[i - 1][j, k])
                            {
                                forwardInputConvolutionOne.Add(forwardOutputConvolution[k]);
                            }
                            //Console.Write((convolutionLinkList[i - 1][j, k] ? 1 : 0) + " ");
                        }
                        forwardInputConvolution.Add(forwardInputConvolutionOne);
                        //Console.WriteLine("");
                    }
                }
                //Console.WriteLine("");
                forwardOutputConvolution = CnnConvolutionLayerList[i].CalculatedResult(forwardInputConvolution);
            }
            //Console.WriteLine("end");
            //计算卷积层转全连接层
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            double[] forwardOutputFull = new double[cnnConvolutionLayerLast.ConvolutionKernelCount
                * cnnConvolutionLayerLast.OutputWidth
                * cnnConvolutionLayerLast.OutputHeight];
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                    {
                        forwardOutputFull[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k] = forwardOutputConvolution[k][i, j];
                    }
                }
            }
            //计算全连接层输出
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                forwardOutputFull = cnnFullLayer.CalculatedResult(forwardOutputFull);
            }
            //double[] outputFullTmp = Predict(input);
            //*/
            forwardOutputFull = Predict(input);
            #endregion

            #region 反向传播
            var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.OutputCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.OutputCount; i++)
            {
                backwardInputFull[i] = output[i] - forwardOutputFull[i];
            }
            //计算全连接层
            /*
            for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            {
                backwardInputFull = CnnFullLayerList[i].BackPropagation(backwardInputFull, learningRate);
                LogHelper.Info(CnnFullLayerList[i].ToString());
                layerCount++;
            }
            //计算全连接层转卷积层
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            List<double[,]> inputConvolutionTmp = new List<double[,]>();
            for (int i = 0; i < cnnConvolutionLayerLast.ConvolutionKernelCount; i++)
            {
                inputConvolutionTmp.Add(new double[cnnConvolutionLayerLast.OutputWidth, cnnConvolutionLayerLast.OutputHeight]);
            }
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                    {
                        inputConvolutionTmp[k][i, j] = backwardInputFull[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k];
                    }
                }
            }
            //计算卷积层
            List<List<double[,]>> backwardInputConvolution = new List<List<double[,]>>();
            for (int i = CnnConvolutionLayerList.Count - 1; i >= 0; i--)
            {
                
                List<double[,]> backwardOutputConvolution = new List<double[,]>();
                if (i == CnnConvolutionLayerList.Count - 1)//最后一层直接输入
                {
                    backwardOutputConvolution = inputConvolutionTmp;
                }
                else//随机链接
                {
                    int[] inputIndex = new int[backwardInputConvolution.Count];//记录当前神经元序号
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)//对应当前层的神经元
                    {
                        double[,] backwardOutputConvolutionOne = new double[backwardInputConvolution[0][0].GetLength(0), backwardInputConvolution[0][0].GetLength(1)];
                        for (int k = 0; k < backwardInputConvolution.Count; k++)//对应下一层的神经元
                        {
                            if (convolutionLinkList[i][k, j])
                            {
                                for (int x = 0; x < backwardInputConvolution[0][0].GetLength(0); x++)
                                {
                                    for (int y = 0; y < backwardInputConvolution[0][0].GetLength(1); y++)
                                    {
                                        backwardOutputConvolutionOne[x, y] += backwardInputConvolution[k][inputIndex[k]][x, y];
                                    }
                                }
                                inputIndex[k]++;
                            }
                            //Console.Write((convolutionLinkList[i][k, j] ? 1 : 0) + " ");
                        }
                        backwardOutputConvolution.Add(backwardOutputConvolutionOne);
                        //Console.WriteLine("");
                    }
                }
                //Console.WriteLine("");
                backwardInputConvolution = CnnConvolutionLayerList[i].BackPropagation(backwardOutputConvolution, learningRate);
                LogHelper.Info(CnnConvolutionLayerList[i].ToString());
                layerCount++;
            }
            //*/
            List<List<double[,]>> backwardInputConvolution = Train(backwardInputFull, learningRate);
            //Console.WriteLine("end");
            #endregion
            return backwardInputConvolution;
        }
        /// <summary>
        /// 训练
        /// </summary>
        public List<List<double[,]>> Train(double[] residual, double learningRate)
        {
            #region 反向传播
            double layerCount = 0;
            var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.OutputCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.OutputCount; i++)
            {
                backwardInputFull[i] = residual[i];
            }
            //计算全连接层
            for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            {
                backwardInputFull = CnnFullLayerList[i].BackPropagation(backwardInputFull, learningRate);
                LogHelper.Info(CnnFullLayerList[i].ToString());
                layerCount++;
            }
            //计算全连接层转卷积层
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            List<double[,]> inputConvolutionTmp = new List<double[,]>();
            for (int i = 0; i < cnnConvolutionLayerLast.ConvolutionKernelCount; i++)
            {
                inputConvolutionTmp.Add(new double[cnnConvolutionLayerLast.OutputWidth, cnnConvolutionLayerLast.OutputHeight]);
            }
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                    {
                        inputConvolutionTmp[k][i, j] = backwardInputFull[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k];
                    }
                }
            }
            //计算卷积层
            List<List<double[,]>> backwardInputConvolution = new List<List<double[,]>>();
            for (int i = CnnConvolutionLayerList.Count - 1; i >= 0; i--)
            {
                List<double[,]> backwardOutputConvolution = new List<double[,]>();
                if (i == CnnConvolutionLayerList.Count - 1)//最后一层直接输入
                {
                    backwardOutputConvolution = inputConvolutionTmp;
                }
                else//随机链接
                {
                    int[] inputIndex = new int[backwardInputConvolution.Count];//记录当前神经元序号
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)//对应当前层的神经元
                    {
                        double[,] backwardOutputConvolutionOne = new double[backwardInputConvolution[0][0].GetLength(0), backwardInputConvolution[0][0].GetLength(1)];
                        for (int k = 0; k < backwardInputConvolution.Count; k++)//对应下一层的神经元
                        {
                            if (convolutionLinkList[i][k, j])
                            {
                                for (int x = 0; x < backwardInputConvolution[0][0].GetLength(0); x++)
                                {
                                    for (int y = 0; y < backwardInputConvolution[0][0].GetLength(1); y++)
                                    {
                                        backwardOutputConvolutionOne[x, y] += backwardInputConvolution[k][inputIndex[k]][x, y];
                                    }
                                }
                                inputIndex[k]++;
                            }
                            //Console.Write((convolutionLinkList[i][k, j] ? 1 : 0) + " ");
                        }
                        backwardOutputConvolution.Add(backwardOutputConvolutionOne);
                        //Console.WriteLine("");
                    }
                }
                //Console.WriteLine("");
                backwardInputConvolution = CnnConvolutionLayerList[i].BackPropagation(backwardOutputConvolution, learningRate);
                LogHelper.Info(CnnConvolutionLayerList[i].ToString());
                layerCount++;
            }
            //Console.WriteLine("end");
            #endregion
            return backwardInputConvolution;
        }
        /// <summary>
        /// 训练,仅用于BP网络
        /// </summary>
        public double[] TrainFullLayer(double[] input, double[] output, double learningRate, ref double[] forwardOutputFull)
        {
            #region 正向传播
            //计算全连接层输出
            forwardOutputFull = input;
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                forwardOutputFull = cnnFullLayer.CalculatedResult(forwardOutputFull);
            }
            #endregion
            #region 反向传播
            var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.OutputCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.OutputCount; i++)
            {
                backwardInputFull[i] = output[i] - forwardOutputFull[i];
            }
            //计算全连接层
            //for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            //{
            //    backwardInputFull = CnnFullLayerList[i].BackPropagation(backwardInputFull, learningRate);
            //}
            backwardInputFull = TrainFullLayer(backwardInputFull, learningRate);
            #endregion
            return backwardInputFull;
        }
        /// <summary>
        /// 训练,仅用于BP网络
        /// </summary>
        public double[] TrainFullLayer(double[] residual, double learningRate)
        {
            #region 反向传播
            var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.OutputCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.OutputCount; i++)
            {
                backwardInputFull[i] = residual[i];
            }
            //计算全连接层
            for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            {
                backwardInputFull = CnnFullLayerList[i].BackPropagation(backwardInputFull, learningRate);
            }
            #endregion
            return backwardInputFull;
        }
        /// <summary>
        /// 识别
        /// </summary>
        public double[] Predict(double[,] input)
        {
            #region 正向传播
            //计算卷积层输出
            List<double[,]> forwardOutputConvolution = new List<double[,]>();//输出值
            for (int i = 0; i < CnnConvolutionLayerList.Count; i++)
            {
                List<List<double[,]>> forwardInputConvolution = new List<List<double[,]>>();//输入值
                if (i == 0)//第一层直接输入
                {
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)
                    {
                        forwardInputConvolution.Add(new List<double[,]> { input });
                    }
                }
                else//随机链接
                {
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)
                    {
                        List<double[,]> forwardInputConvolutionOne = new List<double[,]>();//一个神经元的输入值
                        for (int k = 0; k < forwardOutputConvolution.Count; k++)
                        {
                            if (convolutionLinkList[i - 1][j, k])
                            {
                                forwardInputConvolutionOne.Add(forwardOutputConvolution[k]);
                            }
                        }
                        forwardInputConvolution.Add(forwardInputConvolutionOne);
                    }
                }
                forwardOutputConvolution = CnnConvolutionLayerList[i].CalculatedResult(forwardInputConvolution);
            }
            //计算卷积层转全连接层
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            double[] forwardOutputFull = new double[cnnConvolutionLayerLast.ConvolutionKernelCount
                * cnnConvolutionLayerLast.OutputWidth
                * cnnConvolutionLayerLast.OutputHeight];
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                    {
                        forwardOutputFull[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k] = forwardOutputConvolution[k][i, j];
                    }
                }
            }
            //计算全连接层输出
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                forwardOutputFull = cnnFullLayer.CalculatedResult(forwardOutputFull);
            }
            #endregion
            return forwardOutputFull;
        }
        /// <summary>
        /// 识别,仅用于BP网络
        /// </summary>
        public double[] PredictFullLayer(double[] input)
        {
            //计算全连接层输出
            double[] forwardOutputFull = input;
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                forwardOutputFull = cnnFullLayer.CalculatedResult(forwardOutputFull);
            }
            return forwardOutputFull;
        }
        /// <summary>
        /// 分离ARGB
        /// </summary>
        /// <param name="input"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        /*private double[,] GetARGB(double[,] input, int type)
        {
            double[,] result = new double[input.GetLength(0), input.GetLength(1)];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    int point = Convert.ToInt32(input[i, j]);
                    switch (type)
                    {
                        case 0://B
                            result[i, j] = (point & 0xFF) / 255;
                            break;
                        case 1://G
                            result[i, j] = ((point & 0xFF00) >> 8) / 255d;
                            break;
                        case 2://R
                            result[i, j] = ((point & 0xFF0000) >> 16) / 255d;
                            break;
                        case 3://A
                            result[i, j] = ((point & 0xFF000000) >> 24) / 255d;
                            break;
                    }
                }
            }
            return result;
        }*/
    }
}

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
        public List<CnnLayer> CnnLayerList { get; set; }
        public Cnn()
        {
            CnnLayerList = new List<CnnLayer>();
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
            //创建卷积层
            var tmpCnnConvolutionLayer = new CnnConvolutionLayer(convolutionKernelCount, inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight,
                offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight, activationFunctionType, 1, standardization, null);
            CnnLayerList.Add(tmpCnnConvolutionLayer);

            if (poolingType != 0)
            {
                //创建池化层
                CnnLayerList.Add(new CnnPoolingLayer(convolutionKernelCount,
                    tmpCnnConvolutionLayer.OutputWidth, tmpCnnConvolutionLayer.OutputHeight,
                    poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, activationFunctionType, poolingType));
            }
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
            var cnnLayerLast = CnnLayerList[CnnLayerList.Count - 1];
            int cnnLayerLastWidth = 0;
            int cnnLayerLastHeight = 0;
            int cnnLayerLastCount = 0;
            if (cnnLayerLast.CnnLayerType == CnnLayer.CnnLayerTypeEnum.Convolution)
            {
                //卷积层
                var cnnConvolutionLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnConvolutionLayer;
                cnnLayerLastWidth = cnnConvolutionLayerLast.OutputWidth;
                cnnLayerLastHeight = cnnConvolutionLayerLast.OutputHeight;
                cnnLayerLastCount = cnnConvolutionLayerLast.NeuronCount;
            }
            else if (cnnLayerLast.CnnLayerType == CnnLayer.CnnLayerTypeEnum.Pooling)
            {
                //池化层
                var cnnPoolingLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnPoolingLayer;
                cnnLayerLastWidth = cnnPoolingLayerLast.OutputWidth;
                cnnLayerLastHeight = cnnPoolingLayerLast.OutputHeight;
                cnnLayerLastCount = cnnPoolingLayerLast.NeuronCount;
            }
            bool[,] layerLinks = new bool[convolutionKernelCount, cnnLayerLastCount];
            if (!isFullLayerLinks)
            {
                //随机创建卷积层间连接
                Random random = new Random();
                for (int i = 0; i < convolutionKernelCount; i++)
                {
                    int linkCount = 0;//每层链接数
                    while (linkCount < 2)//确保最低链接数
                    {
                        for (int j = 0; j < cnnLayerLastCount; j++)
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
                        for (int k = 0; k < cnnLayerLastCount; k++)
                        {
                            if (layerLinks[i, k] == layerLinks[j, k])
                                linkCount++;
                        }
                        if (linkCount == cnnLayerLastCount)
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
                    for (int j = 0; j < cnnLayerLastCount; j++)
                    {
                        layerLinks[i, j] = true;
                    }
                }
            }
            //创建卷积层
            var tmpCnnConvolutionLayer = new CnnConvolutionLayer(convolutionKernelCount, cnnLayerLastWidth, cnnLayerLastHeight,
                receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                activationFunctionType, cnnLayerLastCount, standardization, layerLinks);
            CnnLayerList.Add(tmpCnnConvolutionLayer);

            if (poolingType != 0)
            {
                //创建池化层
                CnnLayerList.Add(new CnnPoolingLayer(convolutionKernelCount,
                    tmpCnnConvolutionLayer.OutputWidth, tmpCnnConvolutionLayer.OutputHeight,
                    poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, activationFunctionType, poolingType));
            }
        }
        /// <summary>
        /// 增加全连接层，在卷积层后，要先创建完卷积层
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int outputCount, CnnNeuron.ActivationFunctionTypes activationFunctionType, bool standardization)
        {
            var cnnLayerLast = CnnLayerList[CnnLayerList.Count - 1];
            int inputCount = 0;
            if (cnnLayerLast.CnnLayerType == CnnLayer.CnnLayerTypeEnum.Convolution)
            {
                //卷积层
                var cnnConvolutionLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnConvolutionLayer;
                inputCount = cnnConvolutionLayerLast.NeuronCount
                    * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.OutputHeight;
            }
            else if (cnnLayerLast.CnnLayerType == CnnLayer.CnnLayerTypeEnum.Pooling)
            {
                //池化层
                var cnnPoolingLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnPoolingLayer;
                inputCount = cnnPoolingLayerLast.NeuronCount
                     * cnnPoolingLayerLast.OutputWidth * cnnPoolingLayerLast.OutputHeight;
            }
            else if (cnnLayerLast.CnnLayerType == CnnLayer.CnnLayerTypeEnum.Full)
            {
                //全连接层
                var cnnFullLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnFullLayer;
                inputCount = cnnFullLayerLast.NeuronCount;
            }
            CnnFullLayer cnnFullLayer = new CnnFullLayer(inputCount, outputCount, activationFunctionType, standardization);
            CnnLayerList.Add(cnnFullLayer);
        }
        /// <summary>
        /// 增加全连接层,仅用于BP网络
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int inputCount, int outputCount, CnnNeuron.ActivationFunctionTypes activationFunctionType, bool standardization)
        {
            CnnFullLayer cnnFullLayer = new CnnFullLayer(inputCount,
                outputCount, activationFunctionType, standardization);
            CnnLayerList.Add(cnnFullLayer);
        }
        /// <summary>
        /// 训练
        /// </summary>
        public List<List<double[,]>> Train(double[,] input, double[] output, double learningRate, ref double[] forwardOutputFull)
        {
            #region 正向传播
            forwardOutputFull = Predict(input);
            #endregion

            #region 反向传播
            var cnnFullLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnFullLayer;//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.NeuronCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.NeuronCount; i++)
            {
                backwardInputFull[i] = output[i] - forwardOutputFull[i];
            }
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
            var cnnFullLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnFullLayer;//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.NeuronCount];
            List<double[,]> inputConvolutionTmp = null;
            List<List<double[,]>> backwardInputConvolution = new List<List<double[,]>>();
            bool[,] layerLinksLast = null;//L+1层的链接方式
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.NeuronCount; i++)
            {
                backwardInputFull[i] = residual[i];
            }
            //计算全连接层
            for (int i = CnnLayerList.Count - 1; i >= 0; i--)
            {
                if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Convolution
                    || CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Pooling)
                {
                    int forwardOutputConvolutionCount = 0;
                    int forwardOutputConvolutionWidth = 0;
                    int forwardOutputConvolutionHeight = 0;
                    if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Convolution)
                    {
                        var cnnConvolutionLayerNow = CnnLayerList[i] as CnnConvolutionLayer;
                        forwardOutputConvolutionCount = cnnConvolutionLayerNow.NeuronCount;
                        forwardOutputConvolutionWidth = cnnConvolutionLayerNow.OutputWidth;
                        forwardOutputConvolutionHeight = cnnConvolutionLayerNow.OutputHeight;
                    }
                    else
                    {
                        var cnnPoolingLayerNow = CnnLayerList[i] as CnnPoolingLayer;
                        forwardOutputConvolutionCount = cnnPoolingLayerNow.NeuronCount;
                        forwardOutputConvolutionWidth = cnnPoolingLayerNow.OutputWidth;
                        forwardOutputConvolutionHeight = cnnPoolingLayerNow.OutputHeight;
                    }
                    if (inputConvolutionTmp == null)
                    {
                        inputConvolutionTmp = new List<double[,]>();
                        //计算全连接层转卷积层
                        for (int c = 0; c < forwardOutputConvolutionCount; c++)
                        {
                            inputConvolutionTmp.Add(new double[forwardOutputConvolutionWidth, forwardOutputConvolutionHeight]);
                        }
                        for (int c = 0; c < forwardOutputConvolutionCount; c++)
                        {
                            for (int w = 0; w < forwardOutputConvolutionWidth; w++)
                            {
                                for (int h = 0; h < forwardOutputConvolutionHeight; h++)
                                {
                                    //inputConvolutionTmp[c][w, h] = backwardInputFull[w * forwardOutputConvolutionCount + h * forwardOutputConvolutionWidth * forwardOutputConvolutionCount + c];
                                    inputConvolutionTmp[c][w, h] = backwardInputFull[c * forwardOutputConvolutionWidth * forwardOutputConvolutionHeight + h * forwardOutputConvolutionWidth + w];
                                }
                            }
                        }
                    }
                    List<double[,]> backwardOutputConvolution = new List<double[,]>();
                    if (backwardInputConvolution.Count == 0)//最后一层直接输入
                    {
                        backwardOutputConvolution = inputConvolutionTmp;
                    }
                    else if (layerLinksLast == null)
                    {
                        //L+1层，池化层
                        foreach (var backwardInput in backwardInputConvolution)//一对一连接
                        {
                            backwardOutputConvolution.Add(backwardInput[0]);
                        }
                    }
                    else//随机链接
                    {
                        int[] inputIndex = new int[backwardInputConvolution.Count];//记录当前神经元序号
                        for (int j = 0; j < forwardOutputConvolutionCount; j++)//对应当前层的神经元
                        {
                            double[,] backwardOutputConvolutionOne = new double[backwardInputConvolution[0][0].GetLength(0), backwardInputConvolution[0][0].GetLength(1)];
                            for (int k = 0; k < backwardInputConvolution.Count; k++)//对应下一层的神经元
                            {
                                if (layerLinksLast[k, j])
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
                    if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Convolution)
                    {
                        //计算卷积层残差
                        var cnnConvolutionLayerNow = CnnLayerList[i] as CnnConvolutionLayer;
                        //Console.WriteLine("");
                        backwardInputConvolution = cnnConvolutionLayerNow.BackPropagation(backwardOutputConvolution, learningRate);
                        layerLinksLast = cnnConvolutionLayerNow.LayerLinks;
                    }
                    else if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Pooling)
                    {
                        //计算池化层残差
                        var cnnPoolingLayerNow = CnnLayerList[i] as CnnPoolingLayer;
                        List<double[,]> backwardInputPooling = cnnPoolingLayerNow.BackPropagation(backwardOutputConvolution, learningRate);
                        backwardInputConvolution = new List<List<double[,]>>();
                        foreach (var backwardInputPoolingOne in backwardInputPooling)
                        {
                            backwardInputConvolution.Add(new List<double[,]> { backwardInputPoolingOne });
                        }
                        layerLinksLast = null;
                    }
                }
                else if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Full)
                {
                    //计算全连接层残差
                    var cnnFullLayerNow = CnnLayerList[i] as CnnFullLayer;
                    backwardInputFull = cnnFullLayerNow.BackPropagation(backwardInputFull, learningRate);
                    //LogHelper.Info(backwardInputFull.ToString());
                }
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
            forwardOutputFull = PredictFullLayer(input);
            #endregion
            #region 反向传播
            var cnnFullLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnFullLayer;//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.NeuronCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.NeuronCount; i++)
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
            var cnnFullLayerLast = CnnLayerList[CnnLayerList.Count - 1] as CnnFullLayer;//最后的输出层
            double[] backwardInputFull = new double[cnnFullLayerLast.NeuronCount];
            //计算输出误差
            for (int i = 0; i < cnnFullLayerLast.NeuronCount; i++)
            {
                backwardInputFull[i] = residual[i];
            }
            //计算全连接层
            for (int i = CnnLayerList.Count - 1; i >= 0; i--)
            {
                backwardInputFull = (CnnLayerList[i] as CnnFullLayer).BackPropagation(backwardInputFull, learningRate);
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
            double[] forwardOutputFull = null;
            for (int i = 0; i < CnnLayerList.Count; i++)
            {
                if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Convolution)
                {
                    //计算卷积层输出
                    var cnnConvolutionLayerNow = CnnLayerList[i] as CnnConvolutionLayer;
                    List<List<double[,]>> forwardInputConvolution = new List<List<double[,]>>();//输入值
                    if (i == 0)//第一层直接输入
                    {
                        for (int j = 0; j < cnnConvolutionLayerNow.NeuronCount; j++)
                        {
                            forwardInputConvolution.Add(new List<double[,]> { input });
                        }
                    }
                    else//随机链接
                    {
                        for (int j = 0; j < cnnConvolutionLayerNow.NeuronCount; j++)
                        {
                            List<double[,]> forwardInputConvolutionOne = new List<double[,]>();//一个神经元的输入值
                            for (int k = 0; k < forwardOutputConvolution.Count; k++)
                            {
                                if (cnnConvolutionLayerNow.LayerLinks[j, k])
                                {
                                    forwardInputConvolutionOne.Add(forwardOutputConvolution[k]);
                                }
                            }
                            forwardInputConvolution.Add(forwardInputConvolutionOne);
                        }
                    }
                    forwardOutputConvolution = cnnConvolutionLayerNow.CalculatedResult(forwardInputConvolution);
                }
                else if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Pooling)
                {
                    //计算池化层输出
                    var cnnPoolingLayerNow = CnnLayerList[i] as CnnPoolingLayer;
                    forwardOutputConvolution = cnnPoolingLayerNow.CalculatedResult(forwardOutputConvolution);
                }
                else if (CnnLayerList[i].CnnLayerType == CnnLayer.CnnLayerTypeEnum.Full)
                {
                    //计算全连接层输出
                    var cnnFullLayerNow = CnnLayerList[i] as CnnFullLayer;
                    if (forwardOutputFull == null)
                    {
                        //计算卷积层转全连接层
                        forwardOutputFull = new double[cnnFullLayerNow.InputCount];
                        int forwardOutputConvolutionCount = forwardOutputConvolution.Count;
                        int forwardOutputConvolutionWidth = forwardOutputConvolution[0].GetLength(0);
                        int forwardOutputConvolutionHeight = forwardOutputConvolution[0].GetLength(1);
                        for (int c = 0; c < forwardOutputConvolutionCount; c++)
                        {
                            for (int w = 0; w < forwardOutputConvolutionWidth; w++)
                            {
                                for (int h = 0; h < forwardOutputConvolutionHeight; h++)
                                {
                                    //forwardOutputFull[w * forwardOutputConvolutionCount + h * forwardOutputConvolutionWidth * forwardOutputConvolutionCount + c] = forwardOutputConvolution[c][w, h];
                                    forwardOutputFull[c * forwardOutputConvolutionWidth * forwardOutputConvolutionHeight + h * forwardOutputConvolutionWidth + w] = forwardOutputConvolution[c][w, h];
                                }
                            }
                        }
                    }
                    forwardOutputFull = cnnFullLayerNow.CalculatedResult(forwardOutputFull);
                }
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
            foreach (CnnFullLayer cnnFullLayer in CnnLayerList)
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

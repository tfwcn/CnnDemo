using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
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
        /// 卷积层间连接
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
            int offsetWidth, int offsetHeight, int activationFunctionType,
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, int poolingActivationFunctionType, int poolingType)
        {
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType, 1);
            if (poolingType > 0)//不创建池化层
            {
                //创建池化层
                cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingActivationFunctionType, poolingType);
            }
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
        }
        /// <summary>
        /// 增加后续卷积层
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int activationFunctionType,
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, int poolingActivationFunctionType, int poolingType)
        {
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, cnnConvolutionLayerLast.OutputWidth, cnnConvolutionLayerLast.OutputHeight,
                receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType, cnnConvolutionLayerLast.ConvolutionKernelCount);
            if (poolingType > 0)//不创建池化层
            {
                //创建池化层
                cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingActivationFunctionType, poolingType);
            }
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
            //随机创建卷积层间连接
            bool[,] oneLinks = new bool[convolutionKernelCount, cnnConvolutionLayerLast.ConvolutionKernelCount];
            Random random = new Random();
            for (int i = 0; i < convolutionKernelCount; i++)
            {
                int linkCount = 0;//每层链接数
                while (linkCount < 2)//确保最低链接数
                {
                    for (int j = 0; j < cnnConvolutionLayerLast.ConvolutionKernelCount; j++)
                    {
                        if (random.NextDouble() < 0.5)
                            oneLinks[i, j] = false;
                        else
                        {
                            oneLinks[i, j] = true;
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
                        if (oneLinks[i, k] == oneLinks[j, k])
                            linkCount++;
                    }
                    if (linkCount == cnnConvolutionLayerLast.ConvolutionKernelCount)
                    {
                        i--;
                        break;
                    }
                }
            }
            convolutionLinkList.Add(oneLinks);
        }
        /// <summary>
        /// 增加全连接层，在卷积层后，要先创建完卷积层
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int outputCount, int activationFunctionType = 1)
        {
            if (CnnFullLayerList.Count == 0)
            {
                //连接卷积层
                var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
                CnnFullLayer cnnFullLayer = new CnnFullLayer(cnnConvolutionLayerLast.ConvolutionKernelCount
                    * cnnConvolutionLayerLast.OutputWidth
                    * cnnConvolutionLayerLast.OutputHeight,
                    outputCount, activationFunctionType);
                CnnFullLayerList.Add(cnnFullLayer);
            }
            else
            {
                var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的卷积层
                CnnFullLayer cnnFullLayer = new CnnFullLayer(cnnFullLayerLast.OutputCount,
                    outputCount, activationFunctionType);
                CnnFullLayerList.Add(cnnFullLayer);
            }
        }
        /// <summary>
        /// 增加全连接层,仅用于BP网络
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int inputCount, int outputCount, int activationFunctionType = 1)
        {
            CnnFullLayer cnnFullLayer = new CnnFullLayer(inputCount,
                outputCount, activationFunctionType);
            CnnFullLayerList.Add(cnnFullLayer);
        }
        /// <summary>
        /// 训练
        /// </summary>
        public void Train(double[,] input, double[] output, double learningRate)
        {
            #region 正向传播
            //计算卷积层输出
            List<double[,]> outputConvolutionTmp = new List<double[,]>();
            for (int i = 0; i < CnnConvolutionLayerList[0].ConvolutionKernelCount; i++)
            {
                outputConvolutionTmp.Add(input);
            }
            for (int i = 0; i < CnnConvolutionLayerList.Count; i++)
            {
                List<double[,]> inputTmp = new List<double[,]>();
                if (i == 0)//第一层直接输入
                {
                    inputTmp = outputConvolutionTmp;
                }
                else//随机链接
                {
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)
                    {
                        double[,] inputOneTmp = new double[outputConvolutionTmp[0].GetLength(0), outputConvolutionTmp[0].GetLength(1)];
                        for (int k = 0; k < outputConvolutionTmp.Count; k++)
                        {
                            if (convolutionLinkList[i - 1][j, k])
                            {
                                for (int x = 0; x < outputConvolutionTmp[0].GetLength(0); x++)
                                {
                                    for (int y = 0; y < outputConvolutionTmp[0].GetLength(1); y++)
                                    {
                                        inputOneTmp[x, y] += outputConvolutionTmp[k][x, y];
                                    }
                                }
                            }
                            //Console.Write((convolutionLinkList[i - 1][j, k] ? 1 : 0) + " ");
                        }
                        inputTmp.Add(inputOneTmp);
                        //Console.WriteLine("");
                    }
                }
                //Console.WriteLine("");
                outputConvolutionTmp = CnnConvolutionLayerList[i].CalculatedResult(inputTmp);
            }
            //Console.WriteLine("end");
            //计算卷积层转全连接层
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            double[] outputFullTmp = new double[cnnConvolutionLayerLast.ConvolutionKernelCount
                * cnnConvolutionLayerLast.OutputWidth
                * cnnConvolutionLayerLast.OutputHeight];
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                    {
                        outputFullTmp[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k] = outputConvolutionTmp[k][i, j];
                    }
                }
            }
            //计算全连接层输出
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                outputFullTmp = cnnFullLayer.CalculatedResult(outputFullTmp);
            }
            #endregion
            #region 反向传播
            //计算全连接层
            double[] inputFullTmp = output;
            for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            {
                inputFullTmp = CnnFullLayerList[i].BackPropagation(inputFullTmp, learningRate);
            }
            //计算全连接层转卷积层
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
                        inputConvolutionTmp[k][i, j] = inputFullTmp[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k];
                    }
                }
            }
            //计算卷积层
            for (int i = CnnConvolutionLayerList.Count - 1; i >= 0; i--)
            {
                List<double[,]> outputTmp = new List<double[,]>();
                if (i == CnnConvolutionLayerList.Count - 1)//最后一层直接输入
                {
                    outputTmp = inputConvolutionTmp;
                }
                else//随机链接
                {
                    for (int j = 0; j < CnnConvolutionLayerList[i].ConvolutionKernelCount; j++)
                    {
                        double[,] outputOneTmp = new double[inputConvolutionTmp[0].GetLength(0), inputConvolutionTmp[0].GetLength(1)];
                        for (int k = 0; k < inputConvolutionTmp.Count; k++)
                        {
                            if (convolutionLinkList[i][k, j])
                            {
                                for (int x = 0; x < inputConvolutionTmp[0].GetLength(0); x++)
                                {
                                    for (int y = 0; y < inputConvolutionTmp[0].GetLength(1); y++)
                                    {
                                        outputOneTmp[x, y] += inputConvolutionTmp[k][x, y];
                                    }
                                }
                            }
                            //Console.Write((convolutionLinkList[i][j, k] ? 1 : 0) + " ");
                        }
                        outputTmp.Add(outputOneTmp);
                        //Console.WriteLine("");
                    }
                }
                //Console.WriteLine("");
                inputConvolutionTmp = CnnConvolutionLayerList[i].BackPropagation(outputTmp, learningRate);
            }
            //Console.WriteLine("end");
            #endregion
            CnnHelper.ShowChange(outputFullTmp, output, 60000);
        }
        /// <summary>
        /// 训练,仅用于BP网络
        /// </summary>
        public void TrainFullLayer(double[] input, double[] output, double learningRate)
        {
            #region 正向传播
            //计算全连接层输出
            double[] outputFullTmp = input;
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                outputFullTmp = cnnFullLayer.CalculatedResult(outputFullTmp);
            }
            #endregion
            #region 反向传播
            //计算全连接层
            double[] inputFullTmp = output;
            for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            {
                inputFullTmp = CnnFullLayerList[i].BackPropagation(inputFullTmp, learningRate);
            }
            #endregion
        }
        /// <summary>
        /// 识别
        /// </summary>
        public double[] Predict(double[,] input)
        {
            //计算卷积层输出
            List<double[,]> outputConvolutionTmp = new List<double[,]>();
            for (int i = 0; i < CnnConvolutionLayerList[0].ConvolutionKernelCount; i++)
            {
                outputConvolutionTmp.Add(input);
            }
            foreach (var cnnConvolutionLayer in CnnConvolutionLayerList)
            {
                List<double[,]> inputTmp = new List<double[,]>();
                for (int i = 0; i < cnnConvolutionLayer.ConvolutionKernelCount; i++)
                {
                    inputTmp.Add(outputConvolutionTmp[i % outputConvolutionTmp.Count]);
                }
                outputConvolutionTmp = cnnConvolutionLayer.CalculatedResult(inputTmp);
            }
            //计算全连接层输出
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            double[] outputFullTmp = new double[cnnConvolutionLayerLast.ConvolutionKernelCount
                * cnnConvolutionLayerLast.OutputWidth
                * cnnConvolutionLayerLast.OutputHeight];
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    for (int k = 0; k < cnnConvolutionLayerLast.ConvolutionKernelCount; k++)
                    {
                        outputFullTmp[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k] = outputConvolutionTmp[k][i, j];
                    }
                }
            }
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                outputFullTmp = cnnFullLayer.CalculatedResult(outputFullTmp);
            }
            return outputFullTmp;
        }
        /// <summary>
        /// 识别,仅用于BP网络
        /// </summary>
        public double[] PredictFullLayer(double[] input)
        {
            //计算全连接层输出
            double[] outputFullTmp = input;
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                outputFullTmp = cnnFullLayer.CalculatedResult(outputFullTmp);
            }
            return outputFullTmp;
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

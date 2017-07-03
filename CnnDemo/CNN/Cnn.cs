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
        /// 卷积层间连接
        /// </summary>
        private List<bool[,]> convolutionLinkList;
        /// <summary>
        /// 训练干涉
        /// </summary>
        public delegate bool TrainInterferenceHandler(double[] value);
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
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, int poolingType, bool standardization)
        {
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType, 1, standardization);
            if (poolingType > 0)//不创建池化层
            {
                //创建池化层
                cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingType);
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
            int poolingReceptiveFieldWidth, int poolingReceptiveFieldHeight, int poolingType, bool standardization)
        {
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            //随机创建卷积层间连接
            bool[,] layerLinks = new bool[convolutionKernelCount, cnnConvolutionLayerLast.ConvolutionKernelCount];
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
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, cnnConvolutionLayerLast.OutputWidth, cnnConvolutionLayerLast.OutputHeight,
                receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType, cnnConvolutionLayerLast.ConvolutionKernelCount, standardization, layerLinks);
            if (poolingType > 0)//不创建池化层
            {
                //创建池化层
                cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingType);
            }
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
            //bool[,] oneLinks = {
            //    {true ,true ,true ,false,false,false}, 
            //    {false,true ,true ,true ,false,false},
            //    {false,false,true ,true ,true ,false},
            //    {false,false,false,true ,true ,true },
            //    {true ,false,false,false,true ,true },
            //    {true ,true ,false,false,false,true },

            //    {true ,true ,true ,true ,false,false}, 
            //    {false,true ,true ,true ,true ,false},
            //    {false,false,true ,true ,true ,true },
            //    {true ,false,false,true ,true ,true },
            //    {true ,true ,false,false,true ,true },
            //    {true ,true ,true ,false,false,true },

            //    {true ,true ,false,true ,true ,false},
            //    {false,true ,true ,false,true ,true },
            //    {true ,false,true ,true ,false,true },

            //    {true ,true ,true ,true ,true ,true }
            //};
            convolutionLinkList.Add(layerLinks);
        }
        /// <summary>
        /// 增加全连接层，在卷积层后，要先创建完卷积层
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int outputCount, int activationFunctionType, bool standardization)
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
        public void AddCnnFullLayer(int inputCount, int outputCount, int activationFunctionType, bool standardization)
        {
            CnnFullLayer cnnFullLayer = new CnnFullLayer(inputCount,
                outputCount, activationFunctionType, standardization);
            CnnFullLayerList.Add(cnnFullLayer);
        }
        /// <summary>
        /// 训练
        /// </summary>
        public List<double[,]> Train(double[,] input, double[] output, double learningRate, TrainInterferenceHandler interference)
        {
            #region 正向传播
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
                        forwardOutputFull[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k] = outputConvolutionTmp[k][i, j];
                    }
                }
            }
            //计算全连接层输出
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                forwardOutputFull = cnnFullLayer.CalculatedResult(forwardOutputFull);
            }
            //double[] outputFullTmp = Predict(input);
            #endregion

            CnnHelper.ShowChange(forwardOutputFull, output, 60000);
            //训练干涉
            if (interference != null && interference(forwardOutputFull) == true)
            {
                return null;
            }
            #region 反向传播backward
            double layerCount = 0;
            //计算全连接层
            double[] backwardInputFull = output;
            for (int i = CnnFullLayerList.Count - 1; i >= 0; i--)
            {
                backwardInputFull = CnnFullLayerList[i].BackPropagation(backwardInputFull, learningRate);
                LogHelper.Info(CnnFullLayerList[i].ToString());
                layerCount++;
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
                        inputConvolutionTmp[k][i, j] = backwardInputFull[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount + k];
                    }
                }
            }
            //计算卷积层
            for (int i = CnnConvolutionLayerList.Count - 1; i >= 0; i--)
            {
                List<double[,]> backwardOutputConvolution = new List<double[,]>();
                if (i == CnnConvolutionLayerList.Count - 1)//最后一层直接输入
                {
                    backwardOutputConvolution = inputConvolutionTmp;
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
                            //Console.Write((convolutionLinkList[i][k, j] ? 1 : 0) + " ");
                        }
                        backwardOutputConvolution.Add(outputOneTmp);
                        //Console.WriteLine("");
                    }
                }
                //Console.WriteLine("");
                inputConvolutionTmp = CnnConvolutionLayerList[i].BackPropagation(backwardOutputConvolution, learningRate);
                LogHelper.Info(CnnConvolutionLayerList[i].ToString());
                layerCount++;
            }
            //Console.WriteLine("end");
            #endregion
            return inputConvolutionTmp;
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
            Console.WriteLine("begin");
            for (int j = 0; j < cnnConvolutionLayerLast.OutputHeight; j++)
            {
                for (int i = 0; i < cnnConvolutionLayerLast.OutputWidth; i++)
                {
                    Console.Write(outputFullTmp[i * cnnConvolutionLayerLast.ConvolutionKernelCount + j * cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.ConvolutionKernelCount]);
                }
            }
            Console.WriteLine("end");
            //计算全连接层输出
            foreach (var cnnFullLayer in CnnFullLayerList)
            {
                outputFullTmp = cnnFullLayer.CalculatedResult(outputFullTmp);
            }
            #endregion
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

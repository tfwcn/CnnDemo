using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    public class Cnn
    {
        /// <summary>
        /// 卷积核集合（层数+每层数量）
        /// </summary>
        public List<List<ICnnNode>> CnnKernelList { get; set; }
        /// <summary>
        /// 全链接层
        /// </summary>
        public List<CnnFullLayer> CnnFullList { get; set; }
        /// <summary>
        /// 创建卷积网络
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="subWidth"></param>
        /// <param name="subHeight"></param>
        /// <param name="offsetWidth"></param>
        /// <param name="offsetHeight"></param>
        public void CreateCnn(int width, int height, int subWidth, int subHeight, int offsetWidth, int offsetHeight, int outputCount)
        {
            CnnKernelList = new List<List<ICnnNode>>();
            CnnFullList = new List<CnnFullLayer>();
            //增加第一层,包含4个卷积核，代表4个颜色通道
            CnnKernelList.Add(
                new List<ICnnNode>() {
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight),
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight),
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight),
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight)
                });
            //增加第二层隐藏层,包含5个卷积核，合并4个通道
            CnnKernelList.Add(
                new List<ICnnNode>() {
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1)
                });
            //增加第三层池化层,包含5个卷积核
            CnnKernelList.Add(
                new List<ICnnNode>() {
                    new CnnPooling(CnnKernelList[1][0].ConvolutionKernelWidth,CnnKernelList[1][0].ConvolutionKernelHeight,2,2),
                    new CnnPooling(CnnKernelList[1][0].ConvolutionKernelWidth,CnnKernelList[1][0].ConvolutionKernelHeight,2,2),
                    new CnnPooling(CnnKernelList[1][0].ConvolutionKernelWidth,CnnKernelList[1][0].ConvolutionKernelHeight,2,2),
                    new CnnPooling(CnnKernelList[1][0].ConvolutionKernelWidth,CnnKernelList[1][0].ConvolutionKernelHeight,2,2),
                    new CnnPooling(CnnKernelList[1][0].ConvolutionKernelWidth,CnnKernelList[1][0].ConvolutionKernelHeight,2,2)
                });
            //全连接输出层
            CnnFullList.Add(
                new CnnFullLayer(CnnKernelList[2][0].ConvolutionKernelWidth * CnnKernelList[2][0].ConvolutionKernelHeight * 5, outputCount)
            );
        }
        /// <summary>
        /// 训练
        /// </summary>
        private void Train(double[,] input, double[] output, double learningRate, int count)
        {
            for (int tc = 0; tc < count; tc++)
            {
                #region 正向传播
                //计算第一层输出
                List<double[,]> output1 = new List<double[,]>();
                output1.Add(CnnKernelList[0][0].CalculatedConvolutionResult(GetARGB(input, 0)));
                output1.Add(CnnKernelList[0][1].CalculatedConvolutionResult(GetARGB(input, 1)));
                output1.Add(CnnKernelList[0][2].CalculatedConvolutionResult(GetARGB(input, 2)));
                output1.Add(CnnKernelList[0][3].CalculatedConvolutionResult(GetARGB(input, 3)));
                //计算第二层输出
                double[,] input1 = new double[output1[0].GetLength(0) * 4, output1[0].GetLength(1)];
                for (int j = 0; j < input1.GetLength(1); j++)
                {
                    for (int i = 0; i < input1.GetLength(0); i += 4)
                    {
                        input1[i, j] = output1[0][i / 4, j];
                        input1[i + 1, j] = output1[1][i / 4, j];
                        input1[i + 2, j] = output1[2][i / 4, j];
                        input1[i + 3, j] = output1[3][i / 4, j];
                    }
                }
                List<double[,]> output2 = new List<double[,]>();
                output2.Add(CnnKernelList[1][0].CalculatedConvolutionResult(input1));
                output2.Add(CnnKernelList[1][1].CalculatedConvolutionResult(input1));
                output2.Add(CnnKernelList[1][2].CalculatedConvolutionResult(input1));
                output2.Add(CnnKernelList[1][3].CalculatedConvolutionResult(input1));
                output2.Add(CnnKernelList[1][4].CalculatedConvolutionResult(input1));
                //计算第三层输出
                List<double[,]> output3 = new List<double[,]>();
                output3.Add(CnnKernelList[2][0].CalculatedConvolutionResult(output2[0]));
                output3.Add(CnnKernelList[2][1].CalculatedConvolutionResult(output2[1]));
                output3.Add(CnnKernelList[2][2].CalculatedConvolutionResult(output2[2]));
                output3.Add(CnnKernelList[2][3].CalculatedConvolutionResult(output2[3]));
                output3.Add(CnnKernelList[2][4].CalculatedConvolutionResult(output2[4]));
                //计算第四层输出
                double[] input3 = new double[CnnFullList[0].InputCount];
                for (int j = 0; j < output3[0].GetLength(1); j++)
                {
                    for (int i = 0; i < output3[0].GetLength(0); i++)
                    {
                        input3[i * j * 5] = output3[0][i, j];
                        input3[i * j * 5 + 1] = output3[1][i, j];
                        input3[i * j * 5 + 2] = output3[2][i, j];
                        input3[i * j * 5 + 3] = output3[3][i, j];
                        input3[i * j * 5 + 4] = output3[4][i, j];
                    }
                }
                double[] output4 = CnnFullList[0].CalculatedResult(input3);
                #endregion
                #region 反向传播
                //计算第四层权重
                CnnFullList[0].BackPropagation(input3, output, learningRate);
                //计算第三层权重
                List<double[,]> outputTrue3 = new List<double[,]>();//第三层正确输出
                outputTrue3.Add(new double[output3[0].GetLength(0), output3[0].GetLength(1)]);
                outputTrue3.Add(new double[output3[0].GetLength(0), output3[0].GetLength(1)]);
                outputTrue3.Add(new double[output3[0].GetLength(0), output3[0].GetLength(1)]);
                outputTrue3.Add(new double[output3[0].GetLength(0), output3[0].GetLength(1)]);
                outputTrue3.Add(new double[output3[0].GetLength(0), output3[0].GetLength(1)]);
                for (int i = 0; i < CnnFullList[0].OutputValue.Length; i += 5)
                {
                    output3[0][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i];
                    output3[1][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 1];
                    output3[2][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 2];
                    output3[3][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 3];
                    output3[4][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 4];
                }
                CnnKernelList[2][0].BackPropagation(output2[0], outputTrue3[0], learningRate);
                CnnKernelList[2][1].BackPropagation(output2[1], outputTrue3[1], learningRate);
                CnnKernelList[2][2].BackPropagation(output2[2], outputTrue3[2], learningRate);
                CnnKernelList[2][3].BackPropagation(output2[3], outputTrue3[3], learningRate);
                CnnKernelList[2][4].BackPropagation(output2[4], outputTrue3[4], learningRate);
                //计算第二层权重
                CnnKernelList[1][0].BackPropagation(input1, CnnKernelList[2][0].OutputValue, learningRate);
                CnnKernelList[1][1].BackPropagation(input1, CnnKernelList[2][1].OutputValue, learningRate);
                CnnKernelList[1][2].BackPropagation(input1, CnnKernelList[2][2].OutputValue, learningRate);
                CnnKernelList[1][3].BackPropagation(input1, CnnKernelList[2][3].OutputValue, learningRate);
                CnnKernelList[1][4].BackPropagation(input1, CnnKernelList[2][4].OutputValue, learningRate);
                //计算第一层权重
                /*List<double[,]> outputTrue1 = new List<double[,]>();//第一层正确输出
                outputTrue3.Add(new double[output1[0].GetLength(0), output1[0].GetLength(1)]);
                outputTrue3.Add(new double[output1[0].GetLength(0), output1[0].GetLength(1)]);
                outputTrue3.Add(new double[output1[0].GetLength(0), output1[0].GetLength(1)]);
                outputTrue3.Add(new double[output1[0].GetLength(0), output1[0].GetLength(1)]);
                for (int j = 0; j < output3[0].GetLength(1); j++)
                {
                    for (int i = 0; i < output3[0].GetLength(0); i++)
                    {
                        input3[i * j * 5] = output3[0][i, j];
                        input3[i * j * 5 + 1] = output3[1][i, j];
                        input3[i * j * 5 + 2] = output3[2][i, j];
                        input3[i * j * 5 + 3] = output3[3][i, j];
                        input3[i * j * 5 + 4] = output3[4][i, j];
                    }
                }
                for (int i = 0; i < CnnFullList[0].OutputValue.Length; i += 5)
                {
                    output3[0][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnKernelList[1][0].OutputValue[i];
                    output3[1][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 1];
                    output3[2][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 2];
                    output3[3][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 3];
                    output3[4][i % output3[0].GetLength(0), i / output3[0].GetLength(1)] = CnnFullList[0].OutputValue[i + 4];
                }
                CnnKernelList[0][0].BackPropagation(GetARGB(input, 0), outputTrue1[0], learningRate);
                CnnKernelList[0][1].BackPropagation(GetARGB(input, 1), outputTrue1[1], learningRate);
                CnnKernelList[0][2].BackPropagation(GetARGB(input, 2), outputTrue1[2], learningRate);
                CnnKernelList[0][3].BackPropagation(GetARGB(input, 3), outputTrue1[3], learningRate);*/
                #endregion
            }
        }
        /// <summary>
        /// 识别
        /// </summary>
        public double[] Predict(double[,] input)
        {
            //计算第一层输出
            List<double[,]> output1 = new List<double[,]>();
            output1.Add(CnnKernelList[0][0].CalculatedConvolutionResult(GetARGB(input, 0)));
            output1.Add(CnnKernelList[0][1].CalculatedConvolutionResult(GetARGB(input, 1)));
            output1.Add(CnnKernelList[0][2].CalculatedConvolutionResult(GetARGB(input, 2)));
            output1.Add(CnnKernelList[0][3].CalculatedConvolutionResult(GetARGB(input, 3)));
            //计算第二层输出
            double[,] input1 = new double[output1[0].GetLength(0) * 4, output1[0].GetLength(1)];
            for (int j = 0; j < input1.GetLength(1); j++)
            {
                for (int i = 0; i < input1.GetLength(0); i += 4)
                {
                    input1[i, j] = output1[0][i / 4, j];
                    input1[i + 1, j] = output1[1][i / 4, j];
                    input1[i + 2, j] = output1[2][i / 4, j];
                    input1[i + 3, j] = output1[3][i / 4, j];
                }
            }
            List<double[,]> output2 = new List<double[,]>();
            output2.Add(CnnKernelList[1][0].CalculatedConvolutionResult(input1));
            output2.Add(CnnKernelList[1][1].CalculatedConvolutionResult(input1));
            output2.Add(CnnKernelList[1][2].CalculatedConvolutionResult(input1));
            output2.Add(CnnKernelList[1][3].CalculatedConvolutionResult(input1));
            output2.Add(CnnKernelList[1][4].CalculatedConvolutionResult(input1));
            //计算第三层输出
            List<double[,]> output3 = new List<double[,]>();
            output3.Add(CnnKernelList[2][0].CalculatedConvolutionResult(output2[0]));
            output3.Add(CnnKernelList[2][1].CalculatedConvolutionResult(output2[1]));
            output3.Add(CnnKernelList[2][2].CalculatedConvolutionResult(output2[2]));
            output3.Add(CnnKernelList[2][3].CalculatedConvolutionResult(output2[3]));
            output3.Add(CnnKernelList[2][4].CalculatedConvolutionResult(output2[4]));
            //计算第四层输出
            double[] input3 = new double[CnnFullList[0].InputCount];
            for (int j = 0; j < output3[0].GetLength(1); j++)
            {
                for (int i = 0; i < output3[0].GetLength(0); i++)
                {
                    input3[i * j * 5] = output3[0][i, j];
                    input3[i * j * 5 + 1] = output3[1][i, j];
                    input3[i * j * 5 + 2] = output3[2][i, j];
                    input3[i * j * 5 + 3] = output3[3][i, j];
                    input3[i * j * 5 + 4] = output3[4][i, j];
                }
            }
            double[] output4 = CnnFullList[0].CalculatedResult(input3);
            return output4;
        }
        /// <summary>
        /// 分离ARGB
        /// </summary>
        /// <param name="input"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        private double[,] GetARGB(double[,] input, int type)
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
        }
    }
}

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
        public List<List<CnnKernel>> CnnKernelList { get; set; }
        /// <summary>
        /// 全链接层
        /// </summary>
        public List<CnnFull> CnnFullList { get; set; }
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
            CnnKernelList = new List<List<CnnKernel>>();
            CnnFullList = new List<CnnFull>();
            //增加第一层,包含4个卷积核，代表4个颜色通道
            CnnKernelList.Add(
                new List<CnnKernel>() {
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight),
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight),
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight),
                    new CnnKernel(width,height,subWidth,subHeight,offsetWidth,offsetHeight)
                });
            //增加第二层隐藏层,包含5个卷积核，合并4个通道
            CnnKernelList.Add(
                new List<CnnKernel>() {
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1),
                    new CnnKernel(CnnKernelList[0][0].ConvolutionKernelWidth*4,CnnKernelList[0][0].ConvolutionKernelHeight,4,1,4,1)
                });
            //增加第三层池化层,包含5个卷积核
            CnnKernelList.Add(
                new List<CnnKernel>() {
                    new CnnKernel(CnnKernelList[1][0].ConvolutionKernelWidth*4,CnnKernelList[1][0].ConvolutionKernelHeight,2,2,2,2,2),
                    new CnnKernel(CnnKernelList[1][0].ConvolutionKernelWidth*4,CnnKernelList[1][0].ConvolutionKernelHeight,2,2,2,2,2),
                    new CnnKernel(CnnKernelList[1][0].ConvolutionKernelWidth*4,CnnKernelList[1][0].ConvolutionKernelHeight,2,2,2,2,2),
                    new CnnKernel(CnnKernelList[1][0].ConvolutionKernelWidth*4,CnnKernelList[1][0].ConvolutionKernelHeight,2,2,2,2,2),
                    new CnnKernel(CnnKernelList[1][0].ConvolutionKernelWidth*4,CnnKernelList[1][0].ConvolutionKernelHeight,2,2,2,2,2)
                });
        }
        /// <summary>
        /// 训练
        /// </summary>
        private void Train(double[,] input)
        {
        }
        /// <summary>
        /// 识别
        /// </summary>
        private void Predict(double[,] input)
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
            output3.Add(CnnKernelList[1][0].CalculatedConvolutionResult(output2[0]));
            output3.Add(CnnKernelList[1][1].CalculatedConvolutionResult(output2[1]));
            output3.Add(CnnKernelList[1][2].CalculatedConvolutionResult(output2[2]));
            output3.Add(CnnKernelList[1][3].CalculatedConvolutionResult(output2[3]));
            output3.Add(CnnKernelList[1][4].CalculatedConvolutionResult(output2[4]));
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
                    uint point = Convert.ToUInt32(input[i, j]);
                    switch (type)
                    {
                        case 0://B
                            result[i, j] = point & 0xFF;
                            break;
                        case 1://G
                            result[i, j] = (point & 0xFF00) >> 8;
                            break;
                        case 2://R
                            result[i, j] = (point & 0xFF0000) >> 16;
                            break;
                        case 3://A
                            result[i, j] = (point & 0xFF000000) >> 24;
                            break;
                    }
                }
            }
            return result;
        }
    }
}

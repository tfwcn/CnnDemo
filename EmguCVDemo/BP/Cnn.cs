﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
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
        public Cnn()
        {
            CnnConvolutionLayerList = new List<CnnConvolutionLayer>();
            CnnFullLayerList = new List<CnnFullLayer>();
        }
        /// <summary>
        /// 增加首个卷积层
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int activationFunctionType = 1,
            int poolingReceptiveFieldWidth = 2, int poolingReceptiveFieldHeight = 2, int poolingActivationFunctionType = 1)
        {
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType);
            //创建池化层
            cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingActivationFunctionType);
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
        }
        /// <summary>
        /// 增加后续卷积层
        /// </summary>
        /// <param name="cnnConvolutionLayer"></param>
        public void AddCnnConvolutionLayer(int convolutionKernelCount,
            int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int activationFunctionType = 1,
            int poolingReceptiveFieldWidth = 2, int poolingReceptiveFieldHeight = 2, int poolingActivationFunctionType = 1)
        {
            var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
            CnnConvolutionLayer cnnConvolutionLayer = new CnnConvolutionLayer();
            //创建卷积层
            cnnConvolutionLayer.CreateCnnKernel(convolutionKernelCount, cnnConvolutionLayerLast.OutputWidth, cnnConvolutionLayerLast.OutputHeight,
                receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType);
            //创建池化层
            cnnConvolutionLayer.CreateCnnPooling(poolingReceptiveFieldWidth, poolingReceptiveFieldHeight, poolingActivationFunctionType);
            CnnConvolutionLayerList.Add(cnnConvolutionLayer);
        }
        /// <summary>
        /// 增加全连接层，在卷积层后，要先创建完卷积层
        /// </summary>
        /// <param name="cnnFullLayer"></param>
        public void AddCnnFullLayer(int OutputCount, int activationFunctionType = 1)
        {
            if (CnnFullLayerList.Count == 0)
            {
                //连接卷积层
                var cnnConvolutionLayerLast = CnnConvolutionLayerList[CnnConvolutionLayerList.Count - 1];//最后的卷积层
                CnnFullLayer cnnFullLayer = new CnnFullLayer(cnnConvolutionLayerLast.ConvolutionKernelCount
                    * cnnConvolutionLayerLast.OutputWidth
                    * cnnConvolutionLayerLast.OutputHeight,
                    OutputCount, activationFunctionType);
                CnnFullLayerList.Add(cnnFullLayer);
            }
            else
            {
                var cnnFullLayerLast = CnnFullLayerList[CnnFullLayerList.Count - 1];//最后的卷积层
                CnnFullLayer cnnFullLayer = new CnnFullLayer(cnnFullLayerLast.OutputCount,
                    OutputCount, activationFunctionType);
                CnnFullLayerList.Add(cnnFullLayer);
            }
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
            foreach (var cnnConvolutionLayer in CnnConvolutionLayerList)
            {
                List<double[,]> inputTmp = new List<double[,]>();
                for (int i = 0; i < cnnConvolutionLayer.ConvolutionKernelCount; i++)
                {
                    inputTmp.Add(outputConvolutionTmp[i % outputConvolutionTmp.Count]);
                }
                outputConvolutionTmp = cnnConvolutionLayer.CalculatedResult(inputTmp);
            }
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
                        outputFullTmp[i * j * (cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.OutputHeight) + k] = outputConvolutionTmp[k][i, j];
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
                        inputConvolutionTmp[k][i, j] = inputFullTmp[i * j * (cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.OutputHeight) + k];
                    }
                }
            }
            //计算卷积层
            for (int i = CnnConvolutionLayerList.Count - 1; i >= 0; i--)
            {
                inputConvolutionTmp = CnnConvolutionLayerList[i].BackPropagation(inputConvolutionTmp, learningRate);
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
                        outputFullTmp[i * j * (cnnConvolutionLayerLast.OutputWidth * cnnConvolutionLayerLast.OutputHeight) + k] = outputConvolutionTmp[k][i, j];
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

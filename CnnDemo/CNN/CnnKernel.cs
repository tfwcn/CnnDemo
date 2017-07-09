﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 卷积核
    /// </summary>
    [Serializable]
    public class CnnKernel : CnnNode
    {
        /// <summary>
        /// 共享权重(所有感知野共享),每个输入独立
        /// </summary>
        public List<double[,]> ShareWeight { get; set; }
        /// <summary>
        /// 偏置,每个输入独立
        /// </summary>
        public double OutputOffset { get; set; }
        /// <summary>
        /// 卷积核大小（宽）
        /// </summary>
        public int ConvolutionKernelWidth { get; set; }
        /// <summary>
        /// 卷积核大小（高）
        /// </summary>
        public int ConvolutionKernelHeight { get; set; }
        /// <summary>
        /// 感知野大小（宽）
        /// </summary>
        private int receptiveFieldWidth;
        /// <summary>
        /// 感知野大小（高）
        /// </summary>
        private int receptiveFieldHeight;
        /// <summary>
        /// 输入宽度
        /// </summary>
        private int inputWidth;
        /// <summary>
        /// 输入高度
        /// </summary>
        private int inputHeight;
        /// <summary>
        /// 偏移值（宽）
        /// </summary>
        private int offsetWidth;
        /// <summary>
        /// 偏移值（高）
        /// </summary>
        private int offsetHeight;
        /// <summary>
        /// 感知野偏移值（宽）
        /// </summary>
        private int receptiveFieldOffsetWidth;
        /// <summary>
        /// 感知野偏移值（高）
        /// </summary>
        private int receptiveFieldOffsetHeight;
        /// <summary>
        /// 原输入值
        /// </summary>
        [NonSerialized]
        public List<double[,]> InputValue;
        /// <summary>
        /// 原输出值
        /// </summary>
        [NonSerialized]
        public double[,] OutputValue;
        /// <summary>
        /// 输入数量
        /// </summary>
        public int InputCount { get; set; }
        /// <summary>
        /// 输出数量（该层卷积核数量）
        /// </summary>
        public int OutputCount { get; set; }
        /// <summary>
        /// 输出平均值
        /// </summary>
        private double mean;
        /// <summary>
        /// 输出方差
        /// </summary>
        private double variance;
        /// <summary>
        /// 归一化
        /// </summary>
        public bool Standardization { get; set; }
        /// <summary>
        /// 共享权重梯度集，用于计算平均权重梯度
        /// </summary>
        private List<List<double[,]>> meanListDeltaWeight;
        /// <summary>
        /// 偏置梯度集，用于计算平均偏置梯度
        /// </summary>
        private List<double> meanListDeltaOffset;
        /// <summary>
        /// 平均共享权重梯度
        /// </summary>
        private List<double[,]> meanDeltaWeight;
        /// <summary>
        /// 平均偏置梯度
        /// </summary>
        private double meanDeltaOffset;
        /// <summary>
        /// 平均梯度集上限
        /// </summary>
        private int miniBatchSize = 10;
        /// <summary>
        /// 正则化概率（Dropout）
        /// </summary>
        private double dropoutChance = -1;
        /// <summary>
        /// 正则化状态（Dropout）
        /// </summary>
        private bool dropoutState = false;
        #region 调试参数
        /// <summary>
        /// 输入残差
        /// </summary>
        private double[,] debugResult;
        /// <summary>
        /// 输出残差
        /// </summary>
        private double[,] debugResidual;
        /// <summary>
        /// 权重残差
        /// </summary>
        private double[,] debugDeltaWeight;
        /// <summary>
        /// 偏置残差
        /// </summary>
        private double debugDeltaOffset;
        #endregion

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="offsetWidth"></param>
        /// <param name="offsetHeight"></param>
        /// <param name="activationFunctionType">激活函数类型，1:tanh,2:PReLU,3:Sigmoid</param>
        /// <param name="inputCount"></param>
        /// <param name="outputCount"></param>
        public CnnKernel(int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight,
            int offsetWidth, int offsetHeight, int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            ActivationFunctionTypes activationFunctionType, int inputCount, int outputCount, bool standardization)
        {
            this.receptiveFieldWidth = receptiveFieldWidth;
            this.receptiveFieldHeight = receptiveFieldHeight;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.offsetWidth = offsetWidth;
            this.offsetHeight = offsetHeight;
            this.receptiveFieldOffsetWidth = receptiveFieldOffsetWidth;
            this.receptiveFieldOffsetHeight = receptiveFieldOffsetHeight;
            this.ActivationFunctionType = activationFunctionType;
            this.InputCount = inputCount;
            this.OutputCount = outputCount;
            this.Standardization = standardization;
            //this.ConvolutionKernelWidth = Convert.ToInt32(Math.Ceiling((inputWidth - receptiveFieldWidth) / (double)offsetWidth)) + 1;
            //this.ConvolutionKernelHeight = Convert.ToInt32(Math.Ceiling((inputHeight - receptiveFieldHeight) / (double)offsetHeight)) + 1;
            this.ConvolutionKernelWidth = Convert.ToInt32(Math.Ceiling((inputWidth + offsetWidth - receptiveFieldWidth * receptiveFieldOffsetWidth) / (double)offsetWidth));//卷积核宽
            this.ConvolutionKernelHeight = Convert.ToInt32(Math.Ceiling((inputHeight + offsetHeight - receptiveFieldHeight * receptiveFieldOffsetHeight) / (double)offsetHeight));//卷积核高
            ShareWeight = new List<double[,]>();
            meanDeltaWeight = new List<double[,]>();
            for (int i = 0; i < inputCount; i++)
            {
                ShareWeight.Add(new double[receptiveFieldWidth, receptiveFieldHeight]);
                meanDeltaWeight.Add(new double[receptiveFieldWidth, receptiveFieldHeight]);
            }
            meanListDeltaWeight = new List<List<double[,]>>();
            meanListDeltaOffset = new List<double>();
            InitShareWeight();
        }
        /// <summary>
        /// 前向传播,计算卷积结果
        /// </summary>
        public double[,] CalculatedConvolutionResult(List<double[,]> value)
        {
            InputValue = value;
            double[,] result = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            //for (int i = 0; i < ConvolutionKernelWidth; i++)
            //{
            //    for (int j = 0; j < ConvolutionKernelHeight; j++)
            //    {
            //        for (int k = 0; k < value.Count; k++)
            //        {
            //            result[i, j] += CalculatedConvolutionPointResult(value[k], i, j, k);//卷积
            //        }
            //    }
            //}
            for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
            {
                result = CnnHelper.MatrixAdd(result, CnnHelper.ConvolutionValid(ShareWeight[inputIndex], receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    value[inputIndex], offsetWidth, offsetHeight));//卷积
            }
            if (Standardization)
            {
                mean = CnnHelper.GetMean(result);
                variance = CnnHelper.GetVariance(result, mean);
            }
            //归一化每个结果
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    if (Standardization)
                    {
                        //调用激活函数计算结果
                        double z = (result[i, j] - mean) / Math.Sqrt(variance);
                        //if (double.IsNaN(z))
                        //    z = result[i, j] > 0 ? 1 : -1;
                        result[i, j] = ActivationFunction(z + OutputOffset);
                    }
                    else
                    {
                        result[i, j] = ActivationFunction(result[i, j] + OutputOffset);
                    }
                }
            }
            //正则化
            if (CnnHelper.RandomObj.NextDouble() < dropoutChance)
            {
                for (int i = 0; i < ConvolutionKernelWidth; i++)
                {
                    for (int j = 0; j < ConvolutionKernelHeight; j++)
                    {
                        result[i, j] = 0;
                    }
                }
                dropoutState = true;
            }
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算感知野结果(卷积)
        /// </summary>
        /// <returns></returns>
        private double CalculatedConvolutionPointResult(double[,] value, int x, int y, int index)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < receptiveFieldWidth && offsetWidth * x + i < value.GetLength(0); i++)
            {
                for (int j = 0; j < receptiveFieldHeight && offsetHeight * y + j < value.GetLength(1); j++)
                {
                    result += value[offsetWidth * x + i, offsetHeight * y + j] * ShareWeight[index][i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 初始化共享权重和偏置
        /// </summary>
        private void InitShareWeight()
        {
            Random random = new Random();
            for (int i = 0; i < ShareWeight[0].GetLength(0); i++)
            {
                for (int j = 0; j < ShareWeight[0].GetLength(1); j++)
                {
                    for (int k = 0; k < ShareWeight.Count; k++)
                    {
                        ShareWeight[k][i, j] = GetRandom(random);
                    }
                }
            }
            //OutputOffset = GetRandom(random);
        }
        /// <summary>
        /// 获取随机值
        /// </summary>
        private double GetRandom(Random random)
        {
            double result = 0;
            switch (ActivationFunctionType)
            {
                case ActivationFunctionTypes.ReLU:
                    //PReLU
                    result = random.NextDouble() * 0.0001;
                    break;
                default:
                    result = (random.NextDouble() * 2 - 1) * Math.Sqrt((float)6.0 / (float)(receptiveFieldWidth * receptiveFieldHeight * (InputCount + OutputCount)));
                    break;
            }
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns></returns>
        public List<double[,]> BackPropagation(double[,] output, double learningRate)
        {
            List<double[,]> result = CalculatedBackPropagationResult(output, learningRate);
            return result;
        }
        /// <summary>
        /// 计算反向传播结果
        /// </summary>
        /// <returns></returns>
        private List<double[,]> CalculatedBackPropagationResult(double[,] residual, double learningRate)
        {
            //上一层残差
            List<double[,]> result = new List<double[,]>();
            //当前层残差
            //double[,] residualNow = new double[ConvolutionKernelWidth, ConvolutionKernelHeight];
            //权重残差
            List<double[,]> deltaWeight = new List<double[,]>();
            //偏置残差
            double deltaOffset = 0;
            //正则化
            if (dropoutState)
            {
                for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
                {
                    result.Add(new double[inputWidth, inputHeight]);
                }
                dropoutState = false;
                return result;
            }
            //残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    //residual[i, j] = ActivationFunctionDerivative(OutputValue[i, j]) * (output[i, j] - OutputValue[i, j]);
                    //residual[i, j] = (output[i, j] - OutputValue[i, j]);//正确
                }
            }
            for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
            {
                double[,] tmpResultDelta = CnnHelper.ConvolutionFull(CnnHelper.MatrixRotate180(ShareWeight[inputIndex]), receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    residual, offsetWidth, offsetHeight, inputWidth, inputHeight);
                //double[,] tmpResultDelta = CnnHelper.ConvolutionFull(CnnHelper.MatrixRotate180(ShareWeight[inputIndex]),
                //    residual, inputWidth, inputHeight);//CNN标准
                //double[,] tmpResultDelta = CnnHelper.ConvolutionFull(ShareWeight[inputIndex], residual);//CNN例子
                result.Add(tmpResultDelta);
                //double[,] tmpDeltaWeight = CnnHelper.ConvolutionValid(residual, CnnHelper.MatrixRotate180(InputValue[inputIndex]));//CNN例子
                //double[,] tmpDeltaWeight = CnnHelper.MatrixRotate180(CnnHelper.ConvolutionValid(residual, CnnHelper.MatrixRotate180(InputValue[inputIndex])));//CNN标准
                double[,] tmpDeltaWeight = CnnHelper.MatrixRotate180(CnnHelper.ConvolutionValid(residual, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                    CnnHelper.MatrixRotate180(InputValue[inputIndex]), offsetWidth, offsetHeight));//错
                deltaWeight.Add(tmpDeltaWeight);
            }
            //计算偏置残差
            for (int i = 0; i < ConvolutionKernelWidth; i++)
            {
                for (int j = 0; j < ConvolutionKernelHeight; j++)
                {
                    deltaOffset += residual[i, j];
                }
            }
            //计算平均梯度
            //*
            meanListDeltaWeight.Add(deltaWeight);
            for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
            {
                for (int i = 0; i < receptiveFieldWidth; i++)
                {
                    for (int j = 0; j < receptiveFieldHeight; j++)
                    {
                        if (meanListDeltaWeight.Count > miniBatchSize)
                        {
                            meanDeltaWeight[inputIndex][i, j] -= meanListDeltaWeight[0][inputIndex][i, j] / miniBatchSize;
                            meanDeltaWeight[inputIndex][i, j] += deltaWeight[inputIndex][i, j] / miniBatchSize;
                            meanListDeltaWeight.RemoveAt(0);
                        }
                        else
                        {
                            meanDeltaWeight[inputIndex][i, j] = 0;
                            foreach (var tmpShareWeight in meanListDeltaWeight)
                            {
                                meanDeltaWeight[inputIndex][i, j] += tmpShareWeight[inputIndex][i, j] / meanListDeltaWeight.Count;
                            }
                        }
                    }
                }
            }
            meanListDeltaOffset.Add(deltaOffset);
            if (meanListDeltaOffset.Count > miniBatchSize)
            {
                meanDeltaOffset -= meanListDeltaOffset[0] / miniBatchSize;
                meanDeltaOffset += deltaOffset / miniBatchSize;
                meanListDeltaOffset.RemoveAt(0);
            }
            else
            {
                meanDeltaOffset = 0;
                foreach (var tmpShareOffset in meanListDeltaOffset)
                {
                    meanDeltaOffset += tmpShareOffset / meanListDeltaOffset.Count;
                }
            }
            //*/
            //计算正确输入值
            for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
            {
                for (int i = 0; i < inputWidth; i++)
                {
                    for (int j = 0; j < inputHeight; j++)
                    {
                        //resultDelta[inputIndex][i, j] *= ActivationFunctionDerivative(InputValue[inputIndex][i, j]);
                        if (Standardization)
                        {
                            //反归一化每个结果
                            result[inputIndex][i, j] = result[inputIndex][i, j] * Math.Sqrt(variance) + mean;
                        }
                    }
                }
            }
            //更新权重和偏置
            //UpdateWeight(deltaWeight, learningRate);
            //UpdateOffset(deltaOffset, learningRate);
            UpdateWeight(meanDeltaWeight, learningRate);
            UpdateOffset(meanDeltaOffset, learningRate);
            //调试参数
            //debugResult = resultDelta;
            //debugResidual = residual;
            //debugDeltaWeight = deltaWeight;
            //debugDeltaOffset = deltaOffset;
            return result;
        }
        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="weight">权重</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateWeight(List<double[,]> delta, double learningRate)
        {
            for (int inputIndex = 0; inputIndex < InputCount; inputIndex++)
            {
                for (int i = 0; i < receptiveFieldWidth; i++)
                {
                    for (int j = 0; j < receptiveFieldHeight; j++)
                    {
                        if (ShareWeight.Count != delta.Count
                            || ShareWeight[inputIndex].GetLength(0) != delta[inputIndex].GetLength(0)
                            || ShareWeight[inputIndex].GetLength(1) != delta[inputIndex].GetLength(1))
                            return;
                        ShareWeight[inputIndex][i, j] += learningRate * delta[inputIndex][i, j];// / InputCount;
                    }
                }
            }
        }
        /// <summary>
        /// 更新偏置
        /// </summary>
        /// <param name="weight">偏置</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateOffset(double delta, double learningRate)
        {
            OutputOffset += learningRate * delta;
        }
        /// <summary>
        /// 神经元描述
        /// </summary>
        public override string ToString()
        {
            return String.Format("卷积核:{0}*{1} 感知区:{2}*{3} 输入残差:{4} 输出残差:{5} 权重残差:{6} 偏置残差:{7}",
                ConvolutionKernelWidth, ConvolutionKernelHeight,
                receptiveFieldWidth, receptiveFieldHeight,
                CnnHelper.GetMeanAbs(debugResult), CnnHelper.GetMeanAbs(debugResidual),
                CnnHelper.GetMeanAbs(debugDeltaWeight), debugDeltaOffset);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 全链接层
    /// </summary>
    public class CnnFullLayer
    {
        /// <summary>
        /// 权重
        /// </summary>
        public double[,] InputWeight { get; set; }
        /// <summary>
        /// 偏置
        /// </summary>
        public double[] OutputOffset { get; set; }
        /// <summary>
        /// 输入数量
        /// </summary>
        public int InputCount { get; set; }
        /// <summary>
        /// 输出数量(神经元数量)
        /// </summary>
        public int OutputCount { get; set; }
        /// <summary>
        /// 激活函数类型，1:tanh
        /// </summary>
        private int ActivationFunctionType;
        /// <summary>
        /// 原输入值
        /// </summary>
        public double[] InputValue { get; set; }
        /// <summary>
        /// 原输出值
        /// </summary>
        public double[] OutputValue { get; set; }

        public CnnFullLayer(int InputCount, int OutputCount, int activationFunctionType = 1)
        {
            this.InputCount = InputCount;
            this.OutputCount = OutputCount;
            this.ActivationFunctionType = activationFunctionType;
            InputWeight = new double[OutputCount, InputCount];
            OutputOffset = new double[OutputCount];
            InitInputWeight();
        }
        /// <summary>
        /// 前向传播,计算结果
        /// </summary>
        public double[] CalculatedResult(double[] value)
        {
            InputValue = value;
            double[] result = new double[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                result[i] = CalculatedPointResult(value, i);
            }
            OutputValue = result;
            return result;
        }
        /// <summary>
        /// 计算单个神经元结果
        /// </summary>
        /// <returns></returns>
        private double CalculatedPointResult(double[] value, int index)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < InputCount; i++)
            {
                result += value[i] * InputWeight[index, i];
            }
            result = ActivationFunction(result + OutputOffset[index]);
            return result;
        }
        /// <summary>
        /// 激活函数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunction(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case 1:
                    //tanh
                    result = ActivationFunctionTanh(value);
                    break;
                case 2:
                    //PReLU
                    result = ActivationFunctionPReLU(value);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数导数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionDerivative(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case 1:
                    //tanh
                    result = ActivationFunctionTanhDerivative(value);
                    break;
                case 2:
                    //PReLU
                    result = ActivationFunctionPReLUDerivative(value);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanh(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            result = Math.Tanh(value);
            return result;
        }
        /// <summary>
        /// 激活函数导数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanhDerivative(double value)
        {
            double result = 0;
            //激活函数导数计算结果
            result = 1 - Math.Pow(Math.Tanh(value), 2);//旧
            //result = 1 - Math.Pow(value, 2);
            return result;
        }
        /// <summary>
        /// 激活函数（PReLU）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionPReLU(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            if (value >= 0)
            {
                result = value;
            }
            else
            {
                result = 0.05 * value;
            }
            return result;
        }
        /// <summary>
        /// 激活函数导数（PReLU）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionPReLUDerivative(double value)
        {
            double result = 0;
            //激活函数导数计算结果
            if (value > 0)
            {
                result = 1;
            }
            else if (value == 0)
            {
                result = 0;
            }
            else
            {
                result = 0.05;
            }
            return result;
        }
        /// <summary>
        /// 初始化权重
        /// </summary>
        private void InitInputWeight()
        {
            Random random = new Random();
            for (int i = 0; i < InputWeight.GetLength(0); i++)
            {
                for (int j = 0; j < InputWeight.GetLength(1); j++)
                {
                    InputWeight[i, j] = GetRandom(random);
                }
            }
            for (int i = 0; i < OutputOffset.Length; i++)
            {
                OutputOffset[i] = GetRandom(random);
            }
        }
        /// <summary>
        /// 获取随机值
        /// </summary>
        private double GetRandom(Random random)
        {
            double result = 0;
            switch (ActivationFunctionType)
            {
                case 2:
                    //PReLU
                    result = (random.NextDouble() * Math.Abs(InputCount - OutputCount) + (InputCount > OutputCount ? OutputCount : InputCount)) * Math.Sqrt(InputCount / 2);
                    break;
                default:
                    result = (random.NextDouble() * Math.Abs(InputCount - OutputCount) + (InputCount > OutputCount ? OutputCount : InputCount)) * Math.Sqrt(InputCount);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns>返回更新权重后的输入值</returns>
        public double[] BackPropagation(double[] output, double learningRate)
        {
            double[] result = new double[InputCount];
            //权重残差
            double[,] deltaWeight = new double[OutputCount, InputCount];
            //偏置残差
            double[] deltaOffset = new double[OutputCount];
            //计算上一层的残差
            for (int i = 0; i < OutputCount; i++)
            {
                //残差=导数(输出值)*(输出值-正确值)
               double residual = ActivationFunctionDerivative(OutputValue[i]) * (OutputValue[i] - output[i]);
                for (int j = 0; j < InputCount; j++)
                {
                    //sum(残差)=更新前的权重*残差
                    result[j] += InputWeight[i, j] * residual;
                    //计算权重残差,sum(残差)=残差*输入值
                    deltaWeight[i, j] += residual * InputValue[j];
                }
                deltaOffset[i] = residual;
            }
            for (int i = 0; i < InputCount; i++)
            {
                //上一层的残差=sum(残差)*导数(输入值)
                //result[i] *= ActivationFunctionDerivative(InputValue[i]);//旧代码没有
            }
            //更新权重和偏置
            UpdateWeight(InputWeight, deltaWeight, learningRate);
            UpdateOffset(OutputOffset, deltaOffset, learningRate);
            //计算正确输入值
            for (int i = 0; i < InputCount; i++)
            {
                //正确输入值=旧输入值-sum(残差*更新前的权重)
                result[i] = InputValue[i] - result[i];
            }
            return result;
        }
        /// <summary>
        /// 更新权重
        /// </summary>
        /// <param name="weight">权重</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateWeight(double[,] weight, double[,] delta, double learningRate)
        {
            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    //weight[j, i] -= learningRate * delta[j, i] / (delta[j, i] + 1e-8);
                    weight[j, i] -= learningRate * delta[j, i];//旧
                }
            }
        }
        /// <summary>
        /// 更新偏置
        /// </summary>
        /// <param name="weight">偏置</param>
        /// <param name="delta">残差</param>
        /// <param name="learningRate">学习率</param>
        private void UpdateOffset(double[] offset, double[] delta, double learningRate)
        {
            for (int i = 0; i < OutputCount; i++)
            {
                //offset[i] -= learningRate * delta[i] / (delta[i] + 1e-8);
                offset[i] -= learningRate * delta[i];//旧
            }
        }
    }
}

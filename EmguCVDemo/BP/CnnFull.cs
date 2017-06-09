using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 全链接层
    /// </summary>
    public class CnnFull
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
        public int ActivationFunctionType;

        public CnnFull(int InputCount, int OutputCount, int activationFunctionType = 1)
        {
            this.InputCount = InputCount;
            this.OutputCount = OutputCount;
            this.ActivationFunctionType = activationFunctionType;
            InputWeight = new double[OutputCount, InputCount];
            OutputOffset = new double[OutputCount];
            InitInputWeight();
        }
        /// <summary>
        /// 计算卷积结果
        /// </summary>
        public double[] CalculatedConvolutionResult(double[] value)
        {
            double[] result = new double[OutputCount];
            for (int i = 0; i < OutputCount; i++)
            {
                for (int j = 0; j < InputCount; j++)
                {
                    result[i] = CalculatedConvolutionPointResult(value, j);
                }
            }
            return result;
        }
        /// <summary>
        /// 计算单个神经元结果
        /// </summary>
        /// <returns></returns>
        private double CalculatedConvolutionPointResult(double[] value, int index)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case 1:
                    //tanh
                    result = ActivationFunctionTanh(value, index);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanh(double[] value, int index)
        {
            double result = 0;
            //累计区域内的值
            for (int i = 0; i < InputCount; i++)
            {
                result += value[i] * InputWeight[index, i];
            }
            //调用激活函数计算结果
            result = Math.Tanh(result + OutputOffset[index]);
            return result;
        }
        /// <summary>
        /// 初始化共享权重
        /// </summary>
        private void InitInputWeight()
        {
            Random random = new Random();
            for (int i = 0; i < InputWeight.GetLength(0); i++)
            {
                for (int j = 0; j < InputWeight.GetLength(1); j++)
                {
                    InputWeight[i, j] = random.NextDouble();
                }
            }
            for (int i = 0; i < OutputOffset.Length; i++)
            {
                OutputOffset[i] = random.NextDouble();
            }
        }
    }
}

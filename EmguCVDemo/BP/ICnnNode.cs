using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 卷积核接口，包含卷积核与池化
    /// </summary>
    public interface ICnnNode
    {
        /// <summary>
        /// 卷积核大小（宽）
        /// </summary>
        int ConvolutionKernelWidth { get; set; }
        /// <summary>
        /// 卷积核大小（高）
        /// </summary>
        int ConvolutionKernelHeight { get; set; }
        /// <summary>
        /// 输出值
        /// </summary>
        double[,] OutputValue { get; set; }
        /// <summary>
        /// 计算卷积结果
        /// </summary>
        double[,] CalculatedConvolutionResult(double[,] value);
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="input">上一层的输出值，即该层输入值</param>
        /// <param name="output">正确输出值</param>
        /// <param name="learningRate">学习速率</param>
        /// <returns></returns>
        void BackPropagation(double[,] input, double[,] output, double learningRate);
    }
}

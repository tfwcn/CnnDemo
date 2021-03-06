﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace CnnDemo.CNN
{
    [Serializable]
    public class CnnNode
    {
        /// <summary>
        /// 池化类型
        /// </summary>
        public enum ActivationFunctionTypes
        {
            /// <summary>
            /// Tanh
            /// </summary>
            Tanh = 1,
            /// <summary>
            /// ReLU
            /// </summary>
            ReLU = 2,
            /// <summary>
            /// Sigmoid
            /// </summary>
            Sigmoid = 3
        }
        /// <summary>
        /// 激活函数类型，1:tanh,2:ReLU,3:Sigmoid
        /// </summary>
        public ActivationFunctionTypes ActivationFunctionType { get; set; }
        /// <summary>
        /// 激活函数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        protected double ActivationFunction(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case ActivationFunctionTypes.Tanh:
                    //Tanh
                    result = ActivationFunctionTanh(value);
                    break;
                case ActivationFunctionTypes.ReLU:
                    //ReLU
                    result = ActivationFunctionReLU(value);
                    break;
                case ActivationFunctionTypes.Sigmoid:
                    //Sigmoid
                    result = ActivationFunctionSigmoid(value);
                    break;
            }
            return result;
        }
        /// <summary>
        /// 激活函数导数
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        protected double ActivationFunctionDerivative(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            switch (ActivationFunctionType)
            {
                case ActivationFunctionTypes.Tanh:
                    //tanh
                    result = ActivationFunctionTanhDerivative(value);
                    break;
                case ActivationFunctionTypes.ReLU:
                    //ReLU
                    result = ActivationFunctionReLUDerivative(value);
                    break;
                case ActivationFunctionTypes.Sigmoid:
                    //Sigmoid
                    result = ActivationFunctionSigmoidDerivative(value);
                    break;
            }
            return result;
        }
        //private const double X_STRETCH = 2.0 / 3.0;
        //private const double Y_STRETCH = 1.7159;
        //private const double DERIVATIVE_STRETCH = 4.57573;
        /// <summary>
        /// 激活函数（tanh）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionTanh(double value)
        {
            double result = 0;
            ////调用激活函数计算结果
            result = Math.Tanh(value);
            //result = Y_STRETCH * Math.Tanh(X_STRETCH * value);
            if (double.IsNaN(result) || Double.IsInfinity(result))
                throw new Exception("NaN!");
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
            result = 1 - Math.Pow(value, 2);
            //double coshx = Math.Cosh(X_STRETCH * value);
            //double denominator = Math.Cosh(2.0 * X_STRETCH * value) + 1;
            //result = DERIVATIVE_STRETCH * coshx * coshx / (denominator * denominator);
            if (double.IsNaN(result) || Double.IsInfinity(result))
                throw new Exception("NaN!");
            return result;
        }
        /// <summary>
        /// 激活函数（ReLU）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionReLU(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            if (value >= 0)
            {
                result = value;
            }
            else
            {
                result = 0;
            }
            if (double.IsNaN(result) || Double.IsInfinity(result))
                throw new Exception("NaN!");
            return result;
        }
        /// <summary>
        /// 激活函数导数（ReLU）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionReLUDerivative(double value)
        {
            double result = 0;
            //激活函数导数计算结果
            if (value > 0)
            {
                result = 1;
            }
            else if (value <= 0)
            {
                result = 0;
            }
            if (double.IsNaN(result) || Double.IsInfinity(result))
                throw new Exception("NaN!");
            return result;
        }
        /// <summary>
        /// 激活函数（Sigmoid）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionSigmoid(double value)
        {
            double result = 0;
            //调用激活函数计算结果
            result = 1 / (1 + Math.Pow(Math.E, -value));
            if (double.IsNaN(result) || Double.IsInfinity(result))
                throw new Exception("NaN!");
            return result;
        }
        /// <summary>
        /// 激活函数导数（Sigmoid）
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        private double ActivationFunctionSigmoidDerivative(double value)
        {
            double result = 0;
            //激活函数导数计算结果
            //result = ActivationFunctionSigmoid(value);
            result = value;
            result = result * (1 - result);
            if (double.IsNaN(result) || Double.IsInfinity(result))
                throw new Exception("NaN!");
            return result;
        }
    }
}

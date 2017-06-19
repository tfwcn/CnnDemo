﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    public class CnnNode
    {
        /// <summary>
        /// 激活函数类型，1:tanh,2:池化(Mean Pooling),3:池化(Max Pooling)
        /// </summary>
        public int ActivationFunctionType { get; set; }
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
        protected double ActivationFunctionDerivative(double value)
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
        private const double X_STRETCH = 2.0 / 3.0;
        private const double Y_STRETCH = 1.7159;
        private const double DERIVATIVE_STRETCH = 4.57573;
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
            if (double.IsNaN(result))
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
            if (double.IsNaN(result))
                throw new Exception("NaN!");
            return result;
        }
        /// <summary>
        /// 激活函数（ReLU）
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
                //result = 0.05 * value;
                result = 0;
            }
            if (double.IsNaN(result))
                throw new Exception("NaN!");
            return result;
        }
        /// <summary>
        /// 激活函数导数（ReLU）
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
                //result = 0.05;
                result = 0;
            }
            if (double.IsNaN(result))
                throw new Exception("NaN!");
            return result;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    public class AnnNode
    {
        /// <summary>
        /// 输入权重
        /// </summary>
        public int[] InputWeight { get; set; }
        /// <summary>
        /// 输出权重
        /// </summary>
        public int[] OutputWeight { get; set; }
        public const int TYPE_INPUT = 0;
        public const int TYPE_HIDDEN = 1;
        public const int TYPE_OUTPUT = 2;

        private int type;

        public void setType(int type)
        {
            this.type = type;
        }
        /// <summary>
        /// 激活函数
        /// </summary>
        public int FunType { get; set; }
        // 节点前向输入输出值
        private float mForwardInputValue;
        private float mForwardOutputValue;

        // 节点反向输入输出值
        private float mBackwardInputValue;
        private float mBackwardOutputValue;

        public AnnNode()
        {
        }

        public AnnNode(int type)
        {
            this.type = type;
        }

        /**
         * sigmoid函数，这里用tan-sigmoid，经测试其效果比log-sigmoid好！
         * 
         * @param in
         * @return
         */
        private float forwardSigmoid(float inValue)
        {
            switch (type)
            {
                case TYPE_INPUT:
                    return inValue;
                case TYPE_HIDDEN:
                case TYPE_OUTPUT:
                    return tanhS(inValue);
            }
            return 0;
        }

        /**
         * log-sigmoid函数
         * 
         * @param in
         * @return
         */
        private float logS(float inValue)
        {
            return (float)(1 / (1 + Math.Exp(-inValue)));
        }

        /**
         * log-sigmoid函数的导数
         * 
         * @param in
         * @return
         */
        private float logSDerivative(float inValue)
        {
            return mForwardOutputValue * (1 - mForwardOutputValue) * inValue;
        }

        /**
         * tan-sigmoid函数
         * 
         * @param in
         * @return
         */
        private float tanhS(float inValue)
        {
            return (float)((Math.Exp(inValue) - Math.Exp(-inValue)) / (Math.Exp(inValue) + Math
                    .Exp(-inValue)));
        }

        /**
         * tan-sigmoid函数的导数
         * 
         * @param in
         * @return
         */
        private float tanhSDerivative(float inValue)
        {
            return (float)((1 - Math.Pow(mForwardOutputValue, 2)) * inValue);
        }

        /**
         * 误差反向传播时，激活函数的导数
         * 
         * @param in
         * @return
         */
        private float backwardPropagate(float inValue)
        {
            switch (type)
            {
                case TYPE_INPUT:
                    return inValue;
                case TYPE_HIDDEN:
                case TYPE_OUTPUT:
                    return tanhSDerivative(inValue);
            }
            return 0;
        }

        public float getForwardInputValue()
        {
            return mForwardInputValue;
        }

        public void setForwardInputValue(float mInputValue)
        {
            this.mForwardInputValue = mInputValue;
            setForwardOutputValue(mInputValue);
        }

        public float getForwardOutputValue()
        {
            return mForwardOutputValue;
        }

        private void setForwardOutputValue(float mInputValue)
        {
            this.mForwardOutputValue = forwardSigmoid(mInputValue);
        }

        public float getBackwardInputValue()
        {
            return mBackwardInputValue;
        }

        public void setBackwardInputValue(float mBackwardInputValue)
        {
            this.mBackwardInputValue = mBackwardInputValue;
            setBackwardOutputValue(mBackwardInputValue);
        }

        public float getBackwardOutputValue()
        {
            return mBackwardOutputValue;
        }

        private void setBackwardOutputValue(float input)
        {
            this.mBackwardOutputValue = backwardPropagate(input);
        }

    }
}

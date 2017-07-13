using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN
{
    /// <summary>
    /// 每层神经元
    /// </summary>
    public class CnnLayer
    {
        /// <summary>
        /// 神经元类型
        /// </summary>
        public enum CnnLayerTypeEnum
        {
            Convolution,
            Pooling,
            Full
        }
        /// <summary>
        /// 神经元类型
        /// </summary>
        public CnnLayerTypeEnum CnnLayerType { get; private set; }
        /// <summary>
        /// 神经元数量
        /// </summary>
        public int NeuronCount { get; private set; }

        public CnnLayer(CnnLayerTypeEnum CnnLayerType, int NeuronCount)
        {
            this.CnnLayerType = CnnLayerType;
            this.NeuronCount = NeuronCount;
        }
    }
}

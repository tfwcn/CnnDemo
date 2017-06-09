using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    /// <summary>
    /// 局部感知野
    /// </summary>
    public class CnnSubReceptiveField
    {
        /// <summary>
        /// 共享偏置
        /// </summary>
        public double OutputOffset { get; set; }
        /// <summary>
        /// 输出值
        /// </summary>
        public double OutputValue { get; set; }
    }
}

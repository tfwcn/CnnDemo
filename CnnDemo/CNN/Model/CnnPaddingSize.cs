using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CnnDemo.CNN.Model
{
    /// <summary>
    /// 填充大小
    /// </summary>
    [Serializable]
    public class CnnPaddingSize
    {
        /// <summary>
        /// 左
        /// </summary>
        public int Left { get; set; }
        /// <summary>
        /// 上
        /// </summary>
        public int Top { get; set; }
        /// <summary>
        /// 右
        /// </summary>
        public int Right { get; set; }
        /// <summary>
        /// 下
        /// </summary>
        public int Bottom { get; set; }
        /// <summary>
        /// 设置大小
        /// </summary>
        public void SetSize(int left, int top, int right, int bottom)
        {
            this.Left = left;
            this.Top = top;
            this.Right = right;
            this.Bottom = bottom;
        }
    }
}

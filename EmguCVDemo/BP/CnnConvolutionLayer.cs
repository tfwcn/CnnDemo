using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmguCVDemo.BP
{
    public class CnnConvolutionLayer
    {
        /// <summary>
        /// 卷积层
        /// </summary>
        private List<CnnKernel> CnnKernelList { get; set; }
        /// <summary>
        /// 池化层
        /// </summary>
        private List<CnnPooling> CnnPoolingList { get; set; }
        /// <summary>
        /// 卷积核数量
        /// </summary>
        public int ConvolutionKernelCount { get; set; }
        /// <summary>
        /// 创建卷积层
        /// </summary>
        /// <param name="convolutionKernelCount"></param>
        /// <param name="inputWidth"></param>
        /// <param name="inputHeight"></param>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="offsetWidth"></param>
        /// <param name="offsetHeight"></param>
        /// <param name="activationFunctionType"></param>
        public void CreateCnnKernel(int convolutionKernelCount, int inputWidth, int inputHeight, int receptiveFieldWidth, int receptiveFieldHeight, int offsetWidth, int offsetHeight, int activationFunctionType = 1)
        {
            this.ConvolutionKernelCount = convolutionKernelCount;
            CnnKernelList = new List<CnnKernel>();
            for (int i = 0; i < ConvolutionKernelCount; i++)
            {
                CnnKernelList.Add(new CnnKernel(inputWidth, inputHeight, receptiveFieldWidth, receptiveFieldHeight, offsetWidth, offsetHeight, activationFunctionType));
            }
        }
        /// <summary>
        /// 创建池化层
        /// </summary>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="activationFunctionType"></param>
        public void CreateCnnPooling(int receptiveFieldWidth, int receptiveFieldHeight, int activationFunctionType = 1)
        {
            if (CnnKernelList == null || CnnKernelList.Count == 0)
                throw new Exception("需先创建卷积层");
            CnnPoolingList = new List<CnnPooling>();
            for (int i = 0; i < ConvolutionKernelCount; i++)
            {
                CnnPoolingList.Add(new CnnPooling(CnnKernelList[0].ConvolutionKernelWidth, CnnKernelList[0].ConvolutionKernelHeight, receptiveFieldWidth, receptiveFieldHeight, activationFunctionType));
            }
        }
    }
}

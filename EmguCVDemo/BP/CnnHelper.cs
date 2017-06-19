using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace EmguCVDemo.BP
{
    public class CnnHelper
    {
        public static void ShowChange(double[] output, double[] labels, int inputCount)
        {
            double subVal = 0;
            for (int i = 0; i < output.Length; i++)
            {
                subVal += output[i] - labels[i];
                Console.Write(output[i] + " ");
            }
            Console.WriteLine("");
            Console.WriteLine("CnnChange:" + subVal);
            //均方差
            double mse = 0;
            for (int i = 0; i < output.Length; i++)
            {
                //残差=导数(输出值)*(输出值-正确值)
                mse += Math.Pow(labels[i] - output[i], 2);
            }
            mse = mse / (2 * inputCount);
            Console.WriteLine("MSE:" + mse);
        }

        public static Bitmap GetImg(Cnn cnn)
        {
            int col = cnn.CnnConvolutionLayerList.Count * 2 + cnn.CnnFullLayerList.Count;
            Bitmap img = new Bitmap(1000, 1000);

            return img;
        }
    }
}

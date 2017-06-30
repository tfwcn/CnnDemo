using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace CnnDemo.CNN
{
    public class CnnHelper
    {
        public static int SumCount = 0;
        public static int TrueCount = 0;
        public static double TruePercent = 0;
        public static int LabelsNum = 0;
        public static int ResultNum = 0;

        public static void ShowChange(double[] output, double[] labels, int inputCount)
        {
            //正确率
            SumCount++;
            int outputValue = 0, labelsValue = 0;
            double outputMax = output[0], labelsMax = labels[0];
            for (int i = 0; i < output.Length; i++)
            {
                if (outputMax < output[i])
                {
                    outputValue = i;
                    outputMax = output[i];
                }
                if (labelsMax < labels[i])
                {
                    labelsValue = i;
                    labelsMax = labels[i];
                }
            }
            if (outputValue == labelsValue) TrueCount++;
            ResultNum = outputValue;
            LabelsNum = labelsValue;
            TruePercent = TrueCount / (double)SumCount;
        }
        /// <summary>
        /// 缩放图片
        /// </summary>
        /// <param name="img"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="rects"></param>
        /// <returns></returns>
        public static Bitmap ZoomImg(Bitmap img, int width, int height)
        {
            //缩放图片
            Bitmap tmpZoomImg = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(tmpZoomImg);
            g.Clear(Color.Black);
            int x = 0, y = 0, w = width, h = height;
            double b1, b2;
            b1 = width / (double)height;
            b2 = img.Width / (double)img.Height;
            if (b1 > b2)
            {
                w = (int)(h * b2);
                x = (width - w) / 2;
            }
            else if (b2 > b1)
            {
                h = (int)(w / b2);
                y = (height - h) / 2;
            }
            g.DrawImage(img, x, y, w, h);
            g.Dispose();
            return tmpZoomImg;
        }
        /// <summary>
        /// 卷积操作
        /// </summary>
        /// <param name="receptive">权重集</param>
        /// <param name="value"></param>
        /// <param name="receptiveFieldWidth"></param>
        /// <param name="receptiveFieldHeight"></param>
        /// <param name="offsetWidth"></param>
        /// <param name="offsetHeight"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public static double[,] Convolution(double[,] shareWeight, double[,] value, int offsetWidth, int offsetHeight, int type)
        {
            // 这里的互相关是在后向传播时调用，类似于将Map反转180度再卷积
            // 为了方便计算，这里先将图像扩充一圈
            // 这里的卷积要分成两拨，偶数模板同奇数模板
            int i, j, c, r;
            int halfmapsizew;
            int halfmapsizeh;
            int shareWeightWidth = shareWeight.GetLength(0);
            int shareWeightHeight = shareWeight.GetLength(1);
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            if (shareWeightWidth % 2 == 0 && shareWeightHeight % 2 == 0)
            { // 模板大小为偶数
                halfmapsizew = (shareWeightWidth) / 2; // 卷积模块的半瓣大小
                halfmapsizeh = (shareWeightHeight) / 2;
            }
            else
            {
                halfmapsizew = (shareWeightWidth - 1) / 2; // 卷积模块的半瓣大小
                halfmapsizeh = (shareWeightHeight - 1) / 2;
            }

            // 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1)
            int outSizeW = valueWidth + (shareWeightWidth - 1); // 这里的输出扩大一部分
            int outSizeH = valueHeight + (shareWeightHeight - 1);
            double[,] result = new double[outSizeW, outSizeH];

            // 为了方便计算，将inputData扩大一圈
            double[,] exInputData = MatrixExpand(value, shareWeightWidth - 1, shareWeightHeight - 1);

            for (j = 0; j < outSizeH; j++)
                for (i = 0; i < outSizeW; i++)
                    for (r = 0; r < shareWeightHeight; r++)
                        for (c = 0; c < shareWeightWidth; c++)
                        {
                            result[j, i] = result[j, i] + shareWeight[r, c] * exInputData[j + r, i + c];
                        }

            switch (type)
            { // 根据不同的情况，返回不同的结果
                case 1: // 完全大小的情况
                    return result;
                case 2://输出大小=输入大小
                    double[,] sameres = MatrixCut(result, halfmapsizew, halfmapsizeh);
                    return sameres;
                case 3://卷积大小
                    double[,] validres;
                    if (shareWeightHeight % 2 == 0 && shareWeightWidth % 2 == 0)
                        validres = MatrixCut(result, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
                    else
                        validres = MatrixCut(result, halfmapsizew * 2, halfmapsizeh * 2);
                    return validres;
                default:
                    return result;
            }
        }
        /// <summary>
        /// 卷积操作(扩大)
        /// </summary>
        public static double[,] ConvolutionMax(double[,] shareWeight, double[,] value)
        {
            int shareWeightWidth = shareWeight.GetLength(0);
            int shareWeightHeight = shareWeight.GetLength(1);
            double[,] exInputData = MatrixExpand(value, shareWeightWidth - 1, shareWeightHeight - 1);
            double[,] result = ConvolutionMin(shareWeight, exInputData);
            return result;
        }
        /// <summary>
        /// 卷积操作(缩小)
        /// </summary>
        public static double[,] ConvolutionMin(double[,] shareWeight, double[,] value)
        {
            int shareWeightWidth = shareWeight.GetLength(0);
            int shareWeightHeight = shareWeight.GetLength(1);
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth - (shareWeightWidth - 1), valueHeight - (shareWeightHeight - 1)];
            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    for (int c = 0; c < shareWeightWidth; c++)
                    {
                        for (int r = 0; r < shareWeightHeight; r++)
                        {
                            result[i, j] += shareWeight[c, r] * value[i + c, j + r];
                        }
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// 周围扩展矩阵，补0
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixExpand(double[,] value, int x, int y)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth + x * 2, valueHeight + y * 2];
            for (int j = 0; j < valueHeight + 2 * y; j++)
            {
                for (int i = 0; i < valueWidth + 2 * x; i++)
                {
                    if (j < y || i < x || j >= (valueHeight + y) || i >= (valueWidth + x))
                        result[j, i] = (float)0.0;
                    else
                        result[j, i] = value[j - y, i - x]; // 复制原向量的数据
                }
            }
            return result;
        }
        /// <summary>
        /// 以中心裁剪矩阵
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixCut(double[,] value, int x, int y)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth - x * 2, valueHeight - y * 2];
            for (int j = 0; j < valueHeight; j++)
            {
                for (int i = 0; i < valueWidth; i++)
                {
                    if (j >= y && i >= x && j < (valueHeight - y) && i < (valueWidth - x))
                        result[j - y, i - x] = value[j, i]; // 复制原向量的数据
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵以中心旋转180度
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixRotate180(double[,] value)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth, valueHeight];
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result[valueWidth - i - 1, valueHeight - j - 1] = value[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵克隆
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixClone(double[,] value)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth, valueHeight];
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result[i, j] = value[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 保存网络
        /// </summary>
        /// <param name="cnn"></param>
        /// <param name="path"></param>
        public static void SaveCnn(Cnn cnn, string path)
        {
            FileStream fs = new FileStream(path, FileMode.Create);
            BinaryFormatter bf = new BinaryFormatter();
            bf.Serialize(fs, cnn);
            fs.Close();
        }
        /// <summary>
        /// 加载网络
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static Cnn LoadCnn(string path)
        {
            Cnn cnn = null;
            if (File.Exists(path))
            {
                FileStream fs = new FileStream(path, FileMode.Open);
                BinaryFormatter bf = new BinaryFormatter();
                cnn = bf.Deserialize(fs) as Cnn;
                fs.Close();
            }
            return cnn;
        }
        /// <summary>
        /// 求均值
        /// </summary>
        /// <returns></returns>
        public static double GetMean(double[,] value)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double result = 0;
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result += value[i, j];
                }
            }
            result /= valueWidth * valueHeight;
            return result;
        }
        /// <summary>
        /// 求均值
        /// </summary>
        /// <returns></returns>
        public static double GetMean(double[] value)
        {
            int valueLenght = value.GetLength(0);
            double result = 0;
            for (int i = 0; i < valueLenght; i++)
            {
                result += value[i];
            }
            result /= valueLenght;
            return result;
        }
        /// <summary>
        /// 求方差
        /// </summary>
        /// <returns></returns>
        public static double GetVariance(double[,] value, double mean)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double result = 0;
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result += Math.Pow(value[i, j] - mean, 2);
                }
            }
            result /= valueWidth * valueHeight;
            return result;
        }
        /// <summary>
        /// 求方差
        /// </summary>
        /// <returns></returns>
        public static double GetVariance(double[] value, double mean)
        {
            int valueLenght = value.GetLength(0);
            double result = 0;
            for (int i = 0; i < valueLenght; i++)
            {
                result += Math.Pow(value[i] - mean, 2);
            }
            result /= valueLenght;
            return result;
        }
        /// <summary>
        /// 求绝对值均值
        /// </summary>
        /// <returns></returns>
        public static double GetMeanAbs(double[,] value)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double result = 0;
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result += Math.Abs(value[i, j]);
                }
            }
            result /= valueWidth * valueHeight;
            return result;
        }
        /// <summary>
        /// 求绝对值均值
        /// </summary>
        /// <returns></returns>
        public static double GetMeanAbs(double[] value)
        {
            int valueLenght = value.GetLength(0);
            double result = 0;
            for (int i = 0; i < valueLenght; i++)
            {
                result += Math.Abs(value[i]);
            }
            result /= valueLenght;
            return result;
        }
        /*public delegate void ForeachHandler1(ref int i, ref double value);
        public delegate void ForeachHandler2(ref int i, ref int j, ref double value);
        /// <summary>
        /// 循环二维数组
        /// </summary>
        public static void Foreach(double[,] array, ForeachHandler2 foreachHandler)
        {
            int i = 0, j = 0;
            int w = array.GetLength(0), h = array.GetLength(1);
            foreach (double value in array)
            {
                double tmpValue = value;
                foreachHandler(ref i, ref j, ref tmpValue);
                array[i, j] = tmpValue;
                j++;
                if (j >= h)
                {
                    i++;
                    j = 0;
                }
            }
        }
        /// <summary>
        /// 循环一维数组
        /// </summary>
        public static void Foreach(double[] array, ForeachHandler1 foreachHandler)
        {
            int i = 0;
            int len = array.GetLength(0);
            foreach (double value in array)
            {
                double tmpValue = value;
                foreachHandler(ref i, ref tmpValue);
                array[i] = tmpValue;
                i++;
            }
        }*/
    }
}

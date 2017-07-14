using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using CnnDemo.CNN.Model;

namespace CnnDemo.CNN
{
    public class CnnHelper
    {
        public static int SumCount = 0;
        public static int TrueCount = 0;
        public static double TruePercent = 0;
        public static int LabelsNum = 0;
        public static int ResultNum = 0;
        public static Random RandomObj = new Random();

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
        public static void ShowChange2(double[] output, double[] labels)
        {
            //正确率
            SumCount++;
            if (Math.Abs(output[0] - labels[0]) < 0.5) TrueCount++;
            TruePercent = TrueCount / (double)SumCount;
        }
        #region 图片处理
        /// <summary>
        /// 缩放图片
        /// </summary>
        /// <param name="img"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="rects"></param>
        /// <returns></returns>
        public static Bitmap ImageZoom(Bitmap img, int width, int height)
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
        /// 矩阵变换图片
        /// </summary>
        /// <param name="img"></param>
        /// <param name="transformer">变换矩阵</param>
        /// <returns></returns>
        public static Bitmap ImageTransform(Bitmap img, double[,] transformer)
        {
            //缩放图片
            Bitmap tmpImg = new Bitmap(img.Width, img.Height);
            //图片固定到内存
            BitmapData imgData = img.LockBits(new Rectangle(0, 0, img.Width, img.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            //图片固定到内存
            BitmapData tmpImgData = tmpImg.LockBits(new Rectangle(0, 0, tmpImg.Width, tmpImg.Height), System.Drawing.Imaging.ImageLockMode.WriteOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            int imgBytes = Math.Abs(imgData.Stride) * img.Height;
            int tmpImgBytes = Math.Abs(tmpImgData.Stride) * tmpImg.Height;
            byte[] imgRgbValues = new byte[imgBytes];
            byte[] tmpImgRgbValues = new byte[tmpImgBytes];
            //*
            //读取颜色到数组
            IntPtr imgPtr = imgData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(imgPtr, imgRgbValues, 0, imgBytes);
            for (int w = 0; w < img.Width; w++)
            {
                for (int h = 0; h < img.Height; h++)
                {
                    //变换点坐标
                    double[,] point = new double[1, 3] { { w, h, 1 } };//1列3行
                    double[,] retPoint = CNN.CnnHelper.MatrixMultiplyT(transformer, point);
                    int x = (int)Math.Round(retPoint[0, 0], 0, MidpointRounding.AwayFromZero);
                    int y = (int)Math.Round(retPoint[0, 1], 0, MidpointRounding.AwayFromZero);
                    if (x >= 0 && x < tmpImg.Width
                        && y >= 0 && y < tmpImg.Height)
                    {
                        tmpImgRgbValues[x * tmpImgData.Stride / tmpImg.Width + y * tmpImgData.Stride] = imgRgbValues[w * imgData.Stride / img.Width + h * imgData.Stride];
                        tmpImgRgbValues[x * tmpImgData.Stride / tmpImg.Width + y * tmpImgData.Stride + 1] = imgRgbValues[w * imgData.Stride / img.Width + h * imgData.Stride + 1];
                        tmpImgRgbValues[x * tmpImgData.Stride / tmpImg.Width + y * tmpImgData.Stride + 2] = imgRgbValues[w * imgData.Stride / img.Width + h * imgData.Stride + 2];
                    }
                    else
                    {
                        //tmpImgRgbValues[(int)retPoint[0, 0] * tmpImgData.Stride / tmpImg.Width + (int)retPoint[0, 1] * tmpImgData.Stride] = 0;
                        //tmpImgRgbValues[(int)retPoint[0, 0] * tmpImgData.Stride / tmpImg.Width + (int)retPoint[0, 1] * tmpImgData.Stride + 1] = 0;
                        //tmpImgRgbValues[(int)retPoint[0, 0] * tmpImgData.Stride / tmpImg.Width + (int)retPoint[0, 1] * tmpImgData.Stride + 2] = 0;
                    }
                }
            }
            //复制颜色到图片
            IntPtr tmpImgPtr = tmpImgData.Scan0;
            System.Runtime.InteropServices.Marshal.Copy(tmpImgRgbValues, 0, tmpImgPtr, tmpImgBytes);
            //*/
            img.UnlockBits(imgData);
            tmpImg.UnlockBits(tmpImgData);
            return tmpImg;
        }
        #endregion
        #region 矩阵操作
        /// <summary>
        /// 离散卷积操作(扩大)
        /// </summary>
        public static double[,] ConvolutionFull(double[,] receptiveField, double[,] value,
            int outputWidth, int outputHeight)
        {
            double[,] result = ConvolutionFull(receptiveField, 1, 1,
                value, 1, 1,
                outputWidth, outputHeight);
            return result;
        }
        /// <summary>
        /// 离散卷积操作(扩大)
        /// </summary>
        public static double[,] ConvolutionFull(double[,] receptiveField, int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            double[,] value, int offsetWidth, int offsetHeight,
            int outputWidth, int outputHeight)
        {
            int shareWeightWidth = receptiveField.GetLength(0);
            int shareWeightHeight = receptiveField.GetLength(1);
            double[,] valueScale = MatrixScale(value, offsetWidth * receptiveFieldOffsetWidth, offsetHeight * receptiveFieldOffsetHeight);//放大输入
            valueScale = MatrixExpand(valueScale, shareWeightWidth * receptiveFieldOffsetWidth - 1, shareWeightHeight * receptiveFieldOffsetHeight - 1, 0);//扩充补0
            double[,] valueConvolution = ConvolutionValid(receptiveField, receptiveFieldOffsetWidth, receptiveFieldOffsetHeight,
                valueScale, 1, 1, null);//卷积放大后的输入
            int left = Convert.ToInt32((valueConvolution.GetLength(0) - outputWidth) / 2.0);
            int top = Convert.ToInt32((valueConvolution.GetLength(1) - outputHeight) / 2.0);
            int right = Convert.ToInt32(Math.Ceiling((valueConvolution.GetLength(0) - outputWidth) / 2.0));
            int bottom = Convert.ToInt32(Math.Ceiling((valueConvolution.GetLength(1) - outputHeight) / 2.0));
            double[,] result = MatrixCut(valueConvolution, left, top, right, bottom);//裁剪成输入大小
            return result;
        }
        /// <summary>
        /// 卷积操作(缩小)
        /// </summary>
        public static double[,] ConvolutionValid(double[,] receptiveField, double[,] value)
        {
            double[,] result = ConvolutionValid(receptiveField, 1, 1, value, 1, 1, null);
            return result;
        }
        /// <summary>
        /// 卷积操作(缩小)
        /// </summary>
        public static double[,] ConvolutionValid(double[,] receptiveField, int receptiveFieldOffsetWidth, int receptiveFieldOffsetHeight,
            double[,] value, int offsetWidth, int offsetHeight, CnnPaddingSize paddingSize)
        {
            int shareWeightWidth = receptiveField.GetLength(0);//感知野宽
            int shareWeightHeight = receptiveField.GetLength(1);//感知野高
            int valueWidth = value.GetLength(0);//矩阵宽
            int valueHeight = value.GetLength(1);//矩阵高
            int kernelWidth = Convert.ToInt32(Math.Ceiling((valueWidth + offsetWidth - shareWeightWidth * receptiveFieldOffsetWidth) / (double)offsetWidth));//卷积核宽
            int kernelHeight = Convert.ToInt32(Math.Ceiling((valueHeight + offsetHeight - shareWeightHeight * receptiveFieldOffsetHeight) / (double)offsetHeight));//卷积核高
            int left = Convert.ToInt32((offsetWidth * (kernelWidth - 1) + shareWeightWidth * receptiveFieldOffsetWidth - valueWidth) / 2.0);
            int top = Convert.ToInt32((offsetHeight * (kernelHeight - 1) + shareWeightHeight * receptiveFieldOffsetHeight - valueHeight) / 2.0);
            int right = Convert.ToInt32(Math.Ceiling((offsetWidth * (kernelWidth - 1) + shareWeightWidth * receptiveFieldOffsetWidth - valueWidth) / 2.0));
            int bottom = Convert.ToInt32(Math.Ceiling((offsetHeight * (kernelHeight - 1) + shareWeightHeight * receptiveFieldOffsetHeight - valueHeight) / 2.0));
            if (paddingSize != null)
            {
                paddingSize.SetSize(left, top, right, bottom);
            }
            double[,] valueScale = MatrixExpand(value, left, top, right, bottom, 0);//扩充补0
            double[,] result = new double[kernelWidth, kernelHeight];
            for (int i = 0; i < kernelWidth; i++)
            {
                for (int j = 0; j < kernelHeight; j++)
                {
                    for (int c = 0; c < shareWeightWidth; c++)
                    {
                        for (int r = 0; r < shareWeightHeight; r++)
                        {
                            result[i, j] += receptiveField[c, r] * valueScale[i * offsetWidth + c * receptiveFieldOffsetWidth, j * offsetHeight + r * receptiveFieldOffsetHeight];
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
        public static double[,] MatrixExpand(double[,] value, int x, int y, double defaultValue)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth + x * 2, valueHeight + y * 2];
            for (int i = 0; i < valueWidth + 2 * x; i++)
            {
                for (int j = 0; j < valueHeight + 2 * y; j++)
                {
                    if (j < y || i < x || j >= (valueHeight + y) || i >= (valueWidth + x))
                        result[i, j] = defaultValue;
                    else
                        result[i, j] = value[i - x, j - y]; // 复制原向量的数据
                }
            }
            return result;
        }
        /// <summary>
        /// 周围扩展矩阵，补0
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixExpand(double[,] value, int left, int top, int right, int bottom, double defaultValue)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth + left + right, valueHeight + top + bottom];
            for (int i = 0; i < valueWidth + left + right; i++)
            {
                for (int j = 0; j < valueHeight + top + bottom; j++)
                {
                    if (j < top || i < left || j >= (valueHeight + top) || i >= (valueWidth + left))
                        result[i, j] = defaultValue;
                    else
                        result[i, j] = value[i - left, j - top]; // 复制原向量的数据
                }
            }
            return result;
        }
        /// <summary>
        /// 按倍数缩放矩阵
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixScale(double[,] value, int xScale, int yScale)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth * xScale, valueHeight * yScale];
            for (int i = 0; i < valueWidth * xScale; i++)
            {
                for (int j = 0; j < valueHeight * yScale; j++)
                {
                    result[i, j] = value[i / xScale, j / yScale];
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
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    if (j >= y && i >= x && j < (valueHeight - y) && i < (valueWidth - x))
                        result[i - x, j - y] = value[i, j]; // 复制原向量的数据
                }
            }
            return result;
        }
        /// <summary>
        /// 以中心裁剪矩阵
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixCut(double[,] value, int left, int top, int right, int bottom)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            double[,] result = new double[valueWidth - left - right, valueHeight - top - bottom];
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    if (j >= top && i >= left && j < (valueHeight - bottom) && i < (valueWidth - right))
                        result[i - left, j - top] = value[i, j]; // 复制原向量的数据
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
        /// 矩阵相乘
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixMultiplyT(double[,] value1, double[,] value2)
        {
            int valueCol1 = value1.GetLength(0);
            int valueRow1 = value1.GetLength(1);
            int valueCol2 = value2.GetLength(0);
            int valueRow2 = value2.GetLength(1);
            if (valueCol1 != valueRow2)
                throw new Exception("第一矩阵行数要与第二矩阵列数相等");
            double[,] result = new double[valueCol2, valueRow1];
            for (int i = 0; i < valueCol1; i++)
            {
                for (int j = 0; j < valueRow1; j++)
                {
                    for (int i2 = 0; i2 < valueCol2; i2++)
                    {
                        result[i2, j] += value1[i, j] * value2[i2, i];
                    }
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵相乘
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixMultiply(double[,] value1, double value2)
        {
            int valueWidth = value1.GetLength(0);
            int valueHeight = value1.GetLength(1);
            double[,] result = new double[valueWidth, valueHeight];
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result[i, j] += value1[i, j] * value2;
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵相乘
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixMultiply(double[,] value1, double[,] value2)
        {
            int valueWidth1 = value1.GetLength(0);
            int valueHeight1 = value1.GetLength(1);
            int valueWidth2 = value2.GetLength(0);
            int valueHeight2 = value2.GetLength(1);
            if (valueWidth1 != valueWidth2 || valueHeight1 != valueHeight2)
                throw new Exception("第一矩阵行列数要与第二矩阵行列数相等");
            double[,] result = new double[valueWidth1, valueHeight1];
            for (int i = 0; i < valueWidth1; i++)
            {
                for (int j = 0; j < valueHeight1; j++)
                {
                    result[i, j] += value1[i, j] * value2[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵相加
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixAdd(double[,] value1, double value2)
        {
            int valueWidth = value1.GetLength(0);
            int valueHeight = value1.GetLength(1);
            double[,] result = new double[valueWidth, valueHeight];
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    result[i, j] += value1[i, j] + value2;
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵相加
        /// </summary>
        /// <returns></returns>
        public static double[,] MatrixAdd(double[,] value1, double[,] value2)
        {
            int valueWidth1 = value1.GetLength(0);
            int valueHeight1 = value1.GetLength(1);
            int valueWidth2 = value2.GetLength(0);
            int valueHeight2 = value2.GetLength(1);
            if (valueWidth1 != valueWidth2 || valueHeight1 != valueHeight2)
                throw new Exception("第一矩阵行列数要与第二矩阵行列数相等");
            double[,] result = new double[valueWidth1, valueHeight1];
            for (int i = 0; i < valueWidth1; i++)
            {
                for (int j = 0; j < valueHeight1; j++)
                {
                    result[i, j] += value1[i, j] + value2[i, j];
                }
            }
            return result;
        }
        /// <summary>
        /// 矩阵相加
        /// </summary>
        /// <returns></returns>
        public static double[] MatrixAdd(double[] value1, double[] value2)
        {
            int valueLength1 = value1.GetLength(0);
            int valueLength2 = value2.GetLength(0);
            if (valueLength1 != valueLength2)
                throw new Exception("第一矩阵行列数要与第二矩阵行列数相等");
            double[] result = new double[valueLength1];
            for (int i = 0; i < valueLength1; i++)
            {
                result[i] += value1[i] + value2[i];
            }
            return result;
        }
        #endregion
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
        /// 保存网络
        /// </summary>
        /// <param name="cnn"></param>
        /// <param name="path"></param>
        public static void SaveCnnGroup(CnnGroup cnn, string path)
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
        /// 加载网络
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static CnnGroup LoadCnnGroup(string path)
        {
            CnnGroup cnn = null;
            if (File.Exists(path))
            {
                FileStream fs = new FileStream(path, FileMode.Open);
                BinaryFormatter bf = new BinaryFormatter();
                cnn = bf.Deserialize(fs) as CnnGroup;
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
            if (value == null)
                return 0;
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
            if (value == null)
                return 0;
            int valueLenght = value.GetLength(0);
            double result = 0;
            for (int i = 0; i < valueLenght; i++)
            {
                result += Math.Abs(value[i]);
            }
            result /= valueLenght;
            return result;
        }
        /*
        public delegate void ForeachHandler1(int i);
        public delegate void ForeachHandler2(int i, int j);
        /// <summary>
        /// 循环二维数组
        /// </summary>
        public static void Foreach(double[,] value, ForeachHandler2 foreachHandler)
        {
            int valueWidth = value.GetLength(0);
            int valueHeight = value.GetLength(1);
            for (int i = 0; i < valueWidth; i++)
            {
                for (int j = 0; j < valueHeight; j++)
                {
                    foreachHandler(i, j);
                }
            }
        }
        /// <summary>
        /// 循环一维数组
        /// </summary>
        public static void Foreach(double[] array, ForeachHandler1 foreachHandler)
        {
            int len = array.GetLength(0);
            for (int i = 0; i < len; i++)
            {
                foreachHandler(i);
            }
        }
        //*/
    }
}

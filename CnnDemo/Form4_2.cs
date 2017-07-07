using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using CnnDemo.BP;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Threading;
using CnnDemo.CNN;

namespace CnnDemo
{
    public partial class Form4_2 : Form
    {
        private Thread threadCnn;
        private Cnn cnn;
        private int trainCount = 0;
        private double learningRate;
        public Form4_2()
        {
            InitializeComponent();
        }

        private void Form4_2_Load(object sender, EventArgs e)
        {
            cnn = new Cnn();
            #region LeNet-5 结构
            /*
            cnn.AddCnnConvolutionLayer(6, 32, 32, 5, 5, 1, 1, CnnNode.ActivationFunctionTypes.Tanh,
                2, 2, CnnPooling.PoolingTypes.MaxPooling, false);
            cnn.AddCnnConvolutionLayer(16, 5, 5, 1, 1, CnnNode.ActivationFunctionTypes.Tanh,
                2, 2, CnnPooling.PoolingTypes.MeanPooling, false, false);
            cnn.AddCnnConvolutionLayer(120, 5, 5, 1, 1, CnnNode.ActivationFunctionTypes.Tanh,
                0, 0, CnnPooling.PoolingTypes.None, false, false);
            cnn.AddCnnFullLayer(84, CnnNode.ActivationFunctionTypes.Tanh, false);
            cnn.AddCnnFullLayer(10, CnnNode.ActivationFunctionTypes.Tanh, false);
            //*/
            #endregion
            cnn.AddCnnConvolutionLayer(6, 254 * 2, 252, 5, 5, 1, 1, CnnNode.ActivationFunctionTypes.Tanh,
                2, 2, CnnPooling.PoolingTypes.MaxPooling, false);
            cnn.AddCnnConvolutionLayer(16, 5, 5, 1, 1, CnnNode.ActivationFunctionTypes.Tanh,
                2, 2, CnnPooling.PoolingTypes.MeanPooling, false, false);
            cnn.AddCnnConvolutionLayer(20, 5, 5, 1, 1, CnnNode.ActivationFunctionTypes.Tanh,
                2, 2, CnnPooling.PoolingTypes.MeanPooling, false, false);
            cnn.AddCnnFullLayer(84, CnnNode.ActivationFunctionTypes.Tanh, false);
            cnn.AddCnnFullLayer(1, CnnNode.ActivationFunctionTypes.Tanh, false);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (threadCnn != null && threadCnn.ThreadState == ThreadState.Running)
            {
                return;
            }
            learningRate = (double)numLearningRate.Value;
            threadCnn = new Thread(() =>
            {
                try
                {
                    trainCount = 0;
                    //int retry = 0;
                    string path = "F:\\迅雷下载\\lfw\\";
                    StreamReader sr = new StreamReader(path + "pairsDevTrain.txt");
                    int trueCount = Convert.ToInt32(sr.ReadLine());//正确数
                    string nowLine = sr.ReadLine();//当前行
                    List<string[]> dataList = new List<string[]>();
                    while (!String.IsNullOrEmpty(nowLine))
                    {
                        string[] values = nowLine.Split('\t');
                        string[] values2 = new string[4];
                        if (values.Length == 3)
                        {
                            values2[0] = values[0];
                            values2[1] = values[1];
                            values2[2] = values[0];
                            values2[3] = values[2];
                        }
                        else
                        {
                            values2[0] = values[0];
                            values2[1] = values[1];
                            values2[2] = values[2];
                            values2[3] = values[3];
                        }
                        dataList.Add(values2);
                        nowLine = sr.ReadLine();//当前行
                    }
                    while (CnnHelper.TruePercent != 1)
                    {
                        trainCount++;
                        CnnHelper.SumCount = 0;
                        CnnHelper.TrueCount = 0;
                        #region LFW人脸集
                        List<string[]> randomList = new List<string[]>();
                        foreach (var data in dataList)//随机排序
                        {
                            randomList.Insert(CnnHelper.RandomObj.Next(randomList.Count + 1), data);
                        }
                        foreach (var data in randomList)
                        {
                            string file1 = path + "lfw\\" + data[0] + "\\" + data[0] + "_" + Convert.ToInt32(data[1]).ToString("0000") + ".jpg";
                            string file2 = path + "lfw\\" + data[2] + "\\" + data[2] + "_" + Convert.ToInt32(data[3]).ToString("0000") + ".jpg";
                            using (Bitmap img1 = new Bitmap(file1))
                            {
                                using (Bitmap img2 = new Bitmap(file2))
                                {
                                    double[] labels = new double[1];
                                    if (dataList.IndexOf(data) < trueCount)
                                    {
                                        labels[0] = 1;
                                    }
                                    Image<Bgr, float> trainingData1 = new Image<Bgr, float>(img1);
                                    Image<Bgr, float> trainingData2 = new Image<Bgr, float>(img2);
                                    double[,] input = new double[img1.Width * 2 + 8, img1.Height + 2];
                                    for (int w = 2; w < img1.Width; w++)
                                    {
                                        for (int h = 1; h < img1.Height; h++)
                                        {
                                            input[w, h] = Color.FromArgb(0,
                                                (int)trainingData1.Data[h, w, 2],
                                                (int)trainingData1.Data[h, w, 1],
                                                (int)trainingData1.Data[h, w, 0]
                                                ).ToArgb() / (double)0xFFFFFF;
                                            input[img1.Width + 4 + w, h] = Color.FromArgb(0,
                                                (int)trainingData2.Data[h, w, 2],
                                                (int)trainingData2.Data[h, w, 1],
                                                (int)trainingData2.Data[h, w, 0]
                                                ).ToArgb() / (double)0xFFFFFF;
                                        }
                                    }
                                    double[] forwardOutputFull = null;
                                    cnn.Train(input, labels, learningRate, ref forwardOutputFull);
                                    CnnHelper.ShowChange2(forwardOutputFull, labels);
                                    this.Invoke(new Action(() =>
                                    {
                                        lblInfo.Text = String.Format("训练周期:{0} 训练次数:{1}/{2} 正确率:{3:00.####%}", trainCount, CnnHelper.TrueCount, CnnHelper.SumCount, CnnHelper.TrueCount / (double)CnnHelper.SumCount);

                                        lblResult.Text = labels[0].ToString();
                                        pbImage1.Image = new Bitmap(img1);
                                        pbImage2.Image = new Bitmap(img2);
                                    }));
                                }
                            }
                        }
                        #endregion
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.ToString());
                }
            });
            threadCnn.Start();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (threadCnn != null && threadCnn.ThreadState == ThreadState.Running)
            {
                return;
            }
            threadCnn = new Thread(() =>
            {
                CnnHelper.SumCount = 0;
                CnnHelper.TrueCount = 0;
                #region LFW人脸集
                foreach (var file in Directory.GetFiles("img2", "*.jpg"))
                {
                    using (Bitmap img = new Bitmap(file))
                    {
                        byte label = Convert.ToByte(file.Substring(file.LastIndexOf('\\') + 1, 1));
                        Image<Bgr, float> trainingData = new Image<Bgr, float>(img); ;
                        double[,] input = new double[img.Width, img.Height];
                        for (int w = 0; w < img.Width; w++)
                        {
                            for (int h = 0; h < img.Height; h++)
                            {
                                input[w, h] = Color.FromArgb(0,
                                    (int)trainingData.Data[h, w, 2],
                                    (int)trainingData.Data[h, w, 1],
                                    (int)trainingData.Data[h, w, 0]
                                    ).ToArgb() / (double)0xFFFFFF;
                            }
                        }
                        input = CnnHelper.MatrixExpand(input, 2, 2, 0);
                        double[] labels = cnn.Predict(input);
                        double[] labelsTrue = new double[10];
                        double maxtype = labels[0], max = 0;
                        for (int n = 0; n < 10; n++)
                        {
                            if (maxtype < labels[n])
                            {
                                max = n;
                                maxtype = labels[n];
                            }
                            if (label == n) labelsTrue[n] = 1;
                        }
                        //Console.WriteLine(i + ":" + label + "," + max);
                        CnnHelper.ShowChange(labels, labelsTrue, 10000);
                        this.Invoke(new Action(() =>
                        {
                            lblInfo.Text = String.Format("识别次数:{0}/{1} 正确率:{2:00.####%}", CnnHelper.TrueCount, CnnHelper.SumCount, CnnHelper.TrueCount / (double)CnnHelper.SumCount);

                            lblResult.Text = CnnHelper.LabelsNum + " " + CnnHelper.ResultNum;
                            pbImage1.Image = new Bitmap(img);
                        }));
                    }
                }
                #endregion
            });
            threadCnn.Start();
        }

        private Bitmap GetImage(FileStream fs, int width, int height)
        {
            Bitmap img = new Bitmap(width, height);
            Random random = new Random();
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte[] bytes1 = new byte[1];
                    fs.Read(bytes1, 0, 1);
                    int r, g, b;
                    if (bytes1[0] < 10)
                    {
                        r = bytes1[0] + (byte)random.Next(0, 100);
                        g = bytes1[0] + (byte)random.Next(0, 100);
                        b = bytes1[0] + (byte)random.Next(0, 100);
                    }
                    else
                    {
                        r = bytes1[0];
                        g = bytes1[0];
                        b = bytes1[0];
                    }
                    img.SetPixel(x, y, Color.FromArgb(r, g, b));
                }
            }
            return img;
        }

        private byte GetLable(FileStream fs)
        {
            byte[] bytes1 = new byte[1];
            fs.Read(bytes1, 0, 1);
            return bytes1[0];
        }

        private int ToInt32(byte[] bytes)
        {
            int count = bytes[3];
            count += bytes[2] << 8;
            count += bytes[1] << 16;
            count += bytes[0] << 24;
            return count;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            if (threadCnn != null && threadCnn.ThreadState == ThreadState.Running)
            {
                threadCnn.Abort();
                threadCnn = null;
            }
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "*.cnn|*.cnn";
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                CnnHelper.SaveCnn(cnn, sfd.FileName);
                //CnnHelper.SaveCnnGroup(cnn, sfd.FileName);
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.cnn|*.cnn";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                cnn = CnnHelper.LoadCnn(ofd.FileName);
                //cnn = CnnHelper.LoadCnnGroup(ofd.FileName);
            }
        }

        private void Form4_2_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (threadCnn != null && threadCnn.ThreadState == ThreadState.Running)
            {
                threadCnn.Abort();
            }
        }

        private void btnPic_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.jpg|*.jpg";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                Bitmap img = new Bitmap(ofd.FileName);
                //img = CnnHelper.ZoomImg(img, 28, 28);
                Image<Bgr, float> trainingData = new Image<Bgr, float>(img);
                double[,] input = new double[28, 28];
                for (int w = 0; w < 28; w++)
                {
                    for (int h = 0; h < 28; h++)
                    {
                        input[w, h] = Color.FromArgb(0,
                            (int)trainingData.Data[h, w, 2],
                            (int)trainingData.Data[h, w, 1],
                            (int)trainingData.Data[h, w, 0]
                            ).ToArgb() / (double)0xFFFFFF;
                    }
                }
                input = CnnHelper.MatrixExpand(input, 2, 2, 0);
                double[] labels = cnn.Predict(input);
                double[] labelsTrue = new double[10];
                double maxtype = labels[0], max = 0;
                for (int n = 0; n < 10; n++)
                {
                    if (maxtype < labels[n])
                    {
                        max = n;
                        maxtype = labels[n];
                    }
                    Console.Write(labels[n] + " ");
                }
                Console.WriteLine("");
                //Console.WriteLine(i + ":" + label + "," + max);
                //CnnHelper.ShowChange(labels, labelsTrue, 10000);
                this.Invoke(new Action(() =>
                {
                    lblResult.Text = max.ToString();
                    pbImage1.Image = img;
                }));
            }
        }

        private void btnPicTrain_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.jpg|*.jpg";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                Bitmap img = new Bitmap(ofd.FileName);
                byte label = Convert.ToByte(ofd.FileName.Substring(ofd.FileName.LastIndexOf('\\') + 1, 1));
                double[] labels = new double[10];
                for (int i2 = 0; i2 < 10; i2++)
                {
                    labels[i2] = 0;
                }
                labels[label] = 1;
                Image<Bgr, float> trainingData = new Image<Bgr, float>(img); ;
                double[,] input = new double[img.Width, img.Height];
                for (int w = 0; w < img.Width; w++)
                {
                    for (int h = 0; h < img.Height; h++)
                    {
                        input[w, h] = Color.FromArgb(0,
                            (int)trainingData.Data[h, w, 2],
                            (int)trainingData.Data[h, w, 1],
                            (int)trainingData.Data[h, w, 0]
                            ).ToArgb() / (double)0xFFFFFF;
                    }
                }
                input = CnnHelper.MatrixExpand(input, 2, 2, 0);
                double[] forwardOutputFull = null;
                cnn.Train(input, labels, (double)numLearningRate.Value, ref forwardOutputFull);
            }
        }
    }
}

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using EmguCVDemo.BP;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Threading;

namespace EmguCVDemo
{
    public partial class Form4 : Form
    {
        private Thread threadCnn;
        private Cnn cnn;
        private int trainCount = 0;
        private double learningRate;
        public Form4()
        {
            InitializeComponent();
        }

        private void Form4_Load(object sender, EventArgs e)
        {
            cnn = new Cnn();
            cnn.AddCnnConvolutionLayer(40, 28, 28, 5, 5, 1, 1, 1, 2, 2, 1, 2);
            //cnn.AddCnnConvolutionLayer(40, 5, 5, 1, 1, 1, 2, 2, 1, 1);
            //cnn.AddCnnConvolutionLayer(120, 5, 5, 1, 1, 1, 0, 0, 0, 0);
            cnn.AddCnnFullLayer(100, 1);
            cnn.AddCnnFullLayer(10, 1);
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
                    while (true)
                    {
                        trainCount++;
                        CnnHelper.SumCount = 0;
                        CnnHelper.TrueCount = 0;
                        using (FileStream fs = new FileStream("train-labels.idx1-ubyte", FileMode.Open))
                        {
                            using (FileStream fsImages = new FileStream("train-images.idx3-ubyte", FileMode.Open))
                            {
                                byte[] bytes4 = new byte[4];
                                fsImages.Seek(4, SeekOrigin.Current);
                                fs.Seek(8, SeekOrigin.Current);
                                fsImages.Read(bytes4, 0, 4);
                                int count = ToInt32(bytes4);
                                fsImages.Read(bytes4, 0, 4);
                                int height = ToInt32(bytes4);
                                fsImages.Read(bytes4, 0, 4);
                                int width = ToInt32(bytes4);
                                for (int i = 0; i < count; i++)
                                {
                                    Bitmap img = GetImage(fsImages, width, height);
                                    byte label = GetLable(fs);
                                    double[] labels = new double[10];
                                    for (int i2 = 0; i2 < 10; i2++)
                                    {
                                        labels[i2] = 0;
                                    }
                                    labels[label] = 1;
                                    Image<Bgr, float> trainingData = new Image<Bgr, float>(img); ;
                                    double[,] input = new double[width, height];
                                    for (int w = 0; w < width; w++)
                                    {
                                        for (int h = 0; h < height; h++)
                                        {
                                            input[w, h] = Color.FromArgb(0,
                                                (int)trainingData.Data[h, w, 2],
                                                (int)trainingData.Data[h, w, 1],
                                                (int)trainingData.Data[h, w, 0]
                                                ).ToArgb() / (double)0xFFFFFF;
                                        }
                                    }
                                    //input = CnnHelper.MatrixExpand(input, 2, 2);
                                    cnn.Train(input, labels, learningRate * (1 - CnnHelper.TruePercent * CnnHelper.TruePercent));
                                    this.Invoke(new Action(() =>
                                    {
                                        lblInfo.Text = String.Format("训练周期:{0} 训练次数:{1}/{2} 正确率:{3:00.####%}", trainCount, CnnHelper.TrueCount, CnnHelper.SumCount, CnnHelper.TrueCount / (double)CnnHelper.SumCount);
                                        if (i % 20 == 0)
                                        {
                                            lblResult.Text = CnnHelper.LabelsNum + " " + CnnHelper.ResultNum;
                                            pbImage.Image = img;
                                        }
                                        //if (CnnHelper.LabelsNum != CnnHelper.ResultNum && retry < 5)
                                        //{
                                        //    i--;
                                        //    retry++;
                                        //}
                                        //else
                                        //{
                                        //    retry = 0;
                                        //}
                                    }));
                                    //img.Save("imgs/" + i + "_" + label + ".jpg");
                                }
                                fsImages.Close();
                                fs.Close();
                            }
                        }
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
                using (FileStream fs = new FileStream("t10k-labels.idx1-ubyte", FileMode.Open))
                {
                    using (FileStream fsImages = new FileStream("t10k-images.idx3-ubyte", FileMode.Open))
                    {
                        byte[] bytes4 = new byte[4];
                        fsImages.Seek(4, SeekOrigin.Current);
                        fs.Seek(8, SeekOrigin.Current);
                        fsImages.Read(bytes4, 0, 4);
                        int count = ToInt32(bytes4);
                        fsImages.Read(bytes4, 0, 4);
                        int height = ToInt32(bytes4);
                        fsImages.Read(bytes4, 0, 4);
                        int width = ToInt32(bytes4);
                        for (int i = 0; i < count; i++)
                        {
                            Bitmap img = GetImage(fsImages, width, height);
                            byte label = GetLable(fs);
                            Image<Bgr, float> trainingData = new Image<Bgr, float>(img); ;
                            double[,] input = new double[width, height];
                            for (int w = 0; w < width; w++)
                            {
                                for (int h = 0; h < height; h++)
                                {
                                    input[w, h] = Color.FromArgb(0,
                                        (int)trainingData.Data[h, w, 2],
                                        (int)trainingData.Data[h, w, 1],
                                        (int)trainingData.Data[h, w, 0]
                                        ).ToArgb() / (double)0xFFFFFF;
                                }
                            }
                            //input = CnnHelper.MatrixExpand(input, 2, 2);
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
                                if (i % 20 == 0)
                                {
                                    lblResult.Text = CnnHelper.LabelsNum + " " + CnnHelper.ResultNum;
                                    pbImage.Image = img;
                                }
                            }));
                            //img.Save("imgs/" + i + "_" + label + ".jpg");
                        }
                        fsImages.Close();
                        fs.Close();
                    }
                }
            });
            threadCnn.Start();
        }

        private Bitmap GetImage(FileStream fs, int width, int height)
        {
            Bitmap img = new Bitmap(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte[] bytes1 = new byte[1];
                    fs.Read(bytes1, 0, 1);
                    img.SetPixel(x, y, Color.FromArgb(bytes1[0], bytes1[0], bytes1[0]));
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
            }
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "*.cnn|*.cnn";
            if (sfd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                CnnHelper.SaveCnn(cnn, sfd.FileName);
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.cnn|*.cnn";
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                cnn = CnnHelper.LoadCnn(ofd.FileName);
            }
        }

        private void Form4_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (threadCnn != null && threadCnn.ThreadState == ThreadState.Running)
            {
                threadCnn.Abort();
            }
        }
    }
}

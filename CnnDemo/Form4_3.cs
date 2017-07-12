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
    public partial class Form4_3 : Form
    {
        private Thread threadCnn;
        private Cnn cnn;//识别层
        private Cnn cnnResize;//缩放层
        private int trainCount = 0;
        private double learningRate;
        public Form4_3()
        {
            InitializeComponent();
        }

        private void Form4_3_Load(object sender, EventArgs e)
        {
            //缩放层
            cnnResize = new Cnn();
            cnnResize.AddCnnConvolutionLayer(6, 32, 32, 5, 5, 2, 2, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                2, 2, CnnPoolingNeuron.PoolingTypes.MaxPooling, false);
            cnnResize.AddCnnConvolutionLayer(16, 5, 5, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                2, 2, CnnPoolingNeuron.PoolingTypes.MaxPooling, false, false);
            cnnResize.AddCnnFullLayer(6, CnnNeuron.ActivationFunctionTypes.Tanh, false);
            //识别层
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
            cnn.AddCnnConvolutionLayer(6, 32, 32, 5, 5, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                2, 2, CnnPoolingNeuron.PoolingTypes.MaxPooling, false);
            cnn.AddCnnConvolutionLayer(16, 5, 5, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                2, 2, CnnPoolingNeuron.PoolingTypes.MeanPooling, false, false);
            cnn.AddCnnConvolutionLayer(120, 5, 5, 1, 1, 1, 1, CnnNeuron.ActivationFunctionTypes.Tanh,
                0, 0, CnnPoolingNeuron.PoolingTypes.None, false, false);
            cnn.AddCnnFullLayer(84, CnnNeuron.ActivationFunctionTypes.Tanh, false);
            cnn.AddCnnFullLayer(10, CnnNeuron.ActivationFunctionTypes.Tanh, false);
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
                    while (CnnHelper.TruePercent != 1)
                    {
                        trainCount++;
                        CnnHelper.SumCount = 0;
                        CnnHelper.TrueCount = 0;
                        if (!chkHandwritten.Checked)
                        {
                            #region MNIST库
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
                                        int left = CnnHelper.RandomObj.Next(4), top = CnnHelper.RandomObj.Next(4);
                                        input = CnnHelper.MatrixExpand(input, left, top, 4 - left, 4 - top, 0);
                                        double[] forwardOutputFull = null;
                                        cnn.Train(input, labels, learningRate, ref forwardOutputFull);
                                        CnnHelper.ShowChange(forwardOutputFull, labels, 60000);
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
                                        //img.Save("imgs/" + i + "_" + label + ".jpg", System.Drawing.Imaging.ImageFormat.Jpeg);
                                    }
                                    fsImages.Close();
                                    fs.Close();
                                }
                            }
                            #endregion
                        }
                        else
                        {
                            #region 手写字体集
                            foreach (var file in Directory.GetFiles("img", "*.jpg"))
                            {
                                using (Bitmap img = new Bitmap(file))
                                {
                                    byte label = Convert.ToByte(file.Substring(file.LastIndexOf('\\') + 1, 1));
                                    double[] labels = new double[10];
                                    for (int i2 = 0; i2 < 10; i2++)
                                    {
                                        labels[i2] = 0;
                                    }
                                    labels[label] = 1;
                                    //he
                                    Image<Bgr, float> trainingData = new Image<Bgr, float>(img);
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
                                    int left = CnnHelper.RandomObj.Next(4), top = CnnHelper.RandomObj.Next(4);
                                    input = CnnHelper.MatrixExpand(input, left, top, 4 - left, 4 - top, 0);
                                    double[] forwardOutputFull = null;
                                    cnn.Train(input, labels, learningRate, ref forwardOutputFull);
                                    CnnHelper.ShowChange(forwardOutputFull, labels, 60000);
                                    this.Invoke(new Action(() =>
                                    {
                                        lblInfo.Text = String.Format("训练周期:{0} 训练次数:{1}/{2} 正确率:{3:00.####%}", trainCount, CnnHelper.TrueCount, CnnHelper.SumCount, CnnHelper.TrueCount / (double)CnnHelper.SumCount);

                                        lblResult.Text = CnnHelper.LabelsNum + " " + CnnHelper.ResultNum;
                                        pbImage.Image = new Bitmap(img);
                                    }));
                                }
                            }
                            #endregion
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
                if (!chkHandwritten.Checked)
                {
                    #region MNIST库
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
                    #endregion
                }
                else
                {
                    #region 手写字体集
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
                                pbImage.Image = new Bitmap(img);
                            }));
                        }
                    }
                    #endregion
                }
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

        private void Form4_3_FormClosing(object sender, FormClosingEventArgs e)
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
                    pbImage.Image = img;
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
